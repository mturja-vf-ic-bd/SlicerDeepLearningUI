import glob
import os
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from pathlib import Path
import webbrowser
from PIL import Image

from DeepLearnerLib.CONSTANTS import DEFAULT_FILE_PATHS
from DeepLearnerLib.Asynchrony import Asynchrony
from CheckableComboBox import CheckableComboBox

try:
    import tensorboard
except ImportError:
    slicer.util.pip_install('tensorboard==2.7.0')
    import tensorboard

from tensorboard import program


class DeepLearner(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "PlaneCNN-Training"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "SurfaceLearner"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Md Asadullah Turja (UNC Chapel Hill)"]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """This module enables the user to easily train complex deep learning models 
        (such as ResNet, EfficientNet, CNN, etc.) without the need for any coding. Read more: https://github.com/mturja-vf-ic-bd/SlicerDeepLearningUI"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """dummy"""


#
# DeepLearnerWidget
#

class DeepLearnerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.uiWidget = None
        self.connect_form_layout = None
        self.connect_collapsible_button = None
        self.counter = None
        self.webWidget = None
        self.tb_log = None
        self.w = None
        self._asynchrony = None
        self._finishCallback = None
        self._running = False
        self.logic = None
        self.model = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def modifyUI(self):
        self.connect_collapsible_button = ctk.ctkCollapsibleButton()
        # self.connect_collapsible_button = self.ui.inputsCollapsibleButton
        self.connect_collapsible_button.text = 'Input'
        self.connect_form_layout = qt.QGridLayout(self.connect_collapsible_button)
        # self.connect_form_layout.setColumnStretch(0, 1)
        # self.connect_form_layout.setColumnStretch(1, 1)
        ########## Collapsible Layout Widgets
        self.InputDirLineEdit = qt.QLineEdit()
        self.InputDirPushButton = qt.QPushButton("Load Training Data Directory")
        self.modality_label = qt.QLabel('Modality')
        self.modality_label.alignment = 'AlignLeft'
        self.session_label = qt.QLabel('Session')
        self.session_label.alignment = 'AlignLeft'
        self.modalityComboBox = CheckableComboBox()
        self.modalityComboBox.toolTip = "Modality Combo Box."
        self.sessionComboBox = CheckableComboBox()
        self.sessionComboBox.toolTip = "Session Combo Box"
        self.connect_form_layout.addWidget(self.InputDirLineEdit, 0, 1)
        self.connect_form_layout.addWidget(self.InputDirPushButton, 1, 1)
        self.connect_form_layout.addWidget(self.modality_label, 2, 0)
        self.connect_form_layout.addWidget(self.modalityComboBox, 2, 1)
        self.connect_form_layout.addWidget(self.session_label, 3, 0)
        self.connect_form_layout.addWidget(self.sessionComboBox, 3, 1)
        return self.connect_collapsible_button

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/DeepLearnerLib.ui'))
        input_widget = self.modifyUI()
        self.layout.addWidget(input_widget)
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.uiWidget = uiWidget
        self.logic = DeepLearnerLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Initialize training hyperparameters with default values
        self.ui.CVFoldLineEdit.text = "1"
        self.ui.writeDirLineEdit.text = os.path.join(Path.home(), "DeepUI")
        self.ui.tbPortLineEdit.text = "6010"
        self.ui.tbAddressLineEdit.text = "localhost"
        self.ui.monitorLineEdit.text = "validation/valid_loss"

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.InputDirLineEdit.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.epochSpinBox.valueChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.gPUCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.batchSizeSpinBox.valueChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.learningRateSpinBox.valueChanged.connect(self.updateParameterNodeFromGUI)

        # Buttons
        self.InputDirPushButton.connect('clicked(bool)', self.populateTrainDirectory)
        self.ui.StartTrain.connect('clicked(bool)', self.onApplyButton)
        self.ui.showLogPushButton.connect('clicked(bool)', self.showTBLog)
        # Model radio buttons
        self.ui.DenseNetRadio.toggled.connect(self.processRadioButton)
        self.ui.EfficientNetRadio.toggled.connect(self.processRadioButton)
        self.ui.ResNetRadio.toggled.connect(self.processRadioButton)
        self.ui.SimpleCNNRadio.toggled.connect(self.processRadioButton)
        self.ui.SimpleCNNRadio.checked = True

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        # self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.GetNodeReference("InputVolume"):
        #   firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #   if firstVolumeNode:
        #     self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())
        pass

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
        self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
        # TODO: change update parameter nodes
        self._parameterNode.EndModify(wasModified)

    def populateInputDirectory(self, mode="train"):
        """
        Populates self.inputPath
        """
        dialog = qt.QFileDialog(self.uiWidget)
        dialog.setFileMode(qt.QFileDialog.Directory)
        dialog.setViewMode(qt.QFileDialog.Detail)
        if dialog.exec_():
            directoryPath = dialog.selectedFiles()
            logging.debug("onDirectoryChanged: {}".format(directoryPath))
            if mode == "train":
                self.trainDir = directoryPath[0]
                self.InputDirLineEdit.text = directoryPath[0]
                feat_set = set()
                timepoints = set()
                counter = {}
                for sub in os.listdir(self.trainDir):
                    if not os.path.isdir(os.path.join(self.trainDir, sub)):
                        continue
                    for tp in os.listdir(os.path.join(self.trainDir, sub)):
                        if not os.path.isdir(os.path.join(self.trainDir, sub, tp)):
                            continue
                        timepoints.add(tp)
                        for feat in os.listdir(os.path.join(self.trainDir, sub, tp)):
                            if not os.path.isdir(os.path.join(self.trainDir, sub, tp, feat)):
                                continue
                            feat_set.add(feat)
                            if f"{tp}_{feat}" in counter.keys():
                                counter[f"{tp}_{feat}"] += 1
                            else:
                                counter[f"{tp}_{feat}"] = 1
                            if self.w is None:
                                rep_file = glob.glob(os.path.join(self.trainDir, sub, tp, feat, "*.png"))[0]
                                self.w = Image.open(rep_file).size
                for f in feat_set:
                    self.modalityComboBox.addItem(f)
                for tp in timepoints:
                    self.sessionComboBox.addItem(tp)
                self.counter = counter

    def processInputText(self, text):
        text_lst = text.split(":")[1].strip().split(",")
        text_lst = [t.strip() for t in text_lst]
        return text_lst

    def populateTrainDirectory(self):
        self.populateInputDirectory("train")

    def populateTestDirectory(self):
        self.populateInputDirectory("test")

    def processRadioButton(self):
        if self.ui.DenseNetRadio.isChecked():
            self.model = "densenet"
        elif self.ui.EfficientNetRadio.isChecked():
            self.model = "eff_bn"
        elif self.ui.ResNetRadio.isChecked():
            self.model = "resnet"
        else:
            self.model = "cnn"

    # def processInputFields(self, text):
    #
    def populateDataDirectory(self):
        DEFAULT_FILE_PATHS["TRAIN_DATA_DIR"] = self.InputDirLineEdit.text
        DEFAULT_FILE_PATHS["FEATURE_DIRS"] = self.processInputText(self.modalityComboBox.currentText)
        DEFAULT_FILE_PATHS["TIME_POINTS"] = self.processInputText(self.sessionComboBox.currentText)
        print(DEFAULT_FILE_PATHS)
        return DEFAULT_FILE_PATHS, True

    def checkOutputDirectory(self):
        """
        Checks if the output directory is empty. If not raises error.
        If no such directory exists, creates a new one.
        """
        output_path = os.path.join(self.ui.writeDirLineEdit.text, "logs", self.model)
        if os.path.exists(output_path):
            n_files = len(os.listdir(output_path))
            if n_files != 0:
                return False
            return True
        else:
            os.makedirs(output_path)
            return True

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        try:
            # Compute output
            file_paths, enough_samples = self.populateDataDirectory()
            isEmpty = self.checkOutputDirectory()

            if isEmpty and enough_samples:
                print("-" * 47)
                print("-" * 10, "Starting Training Process", "-" * 10)
                print("-" * 47)
                # Disable training buttons
                self.ui.StartTrain.enabled = False
                self.InputDirPushButton.enabled = False
                self.ui.trainingProgressBar.setValue(0)
                feat_dim = len(DEFAULT_FILE_PATHS["FEATURE_DIRS"]) * len(DEFAULT_FILE_PATHS["TIME_POINTS"]) * 2
                self._asynchrony = Asynchrony(
                        lambda: self.logic.process(
                                in_channels=feat_dim,
                                num_classes=2,
                                model_name=self.model,
                                batch_size=int(self.ui.batchSizeSpinBox.value),
                                learning_rate=float(self.ui.learningRateSpinBox.value),
                                max_epochs=int(self.ui.epochSpinBox.value),
                                n_folds=int(self.ui.CVFoldLineEdit.text),
                                use_gpu=self.ui.gPUCheckBox.isChecked(),
                                logdir=self.ui.writeDirLineEdit.text,
                                exp_name="default",
                                cp_n_epoch=int(self.ui.cpFreqSpinBox.value),
                                max_cp=int(self.ui.maxCpSpinBox.value),
                                monitor=self.ui.monitorLineEdit.text,
                                pos_weight=float(self.ui.posWLineEdit.text),
                                file_paths=file_paths,
                                ui=self.ui,
                                w=self.w
                            )
                )
                self._asynchrony.Start()
                self._running = True

            elif not enough_samples:
                self.ui.sample_error_dialog = qt.QErrorMessage()
                self.ui.sample_error_dialog.showMessage('Not enough samples!')
            elif not isEmpty:
                self.ui.error_dialog = qt.QErrorMessage()
                self.ui.error_dialog.showMessage('Output directory not empty!')
        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: " + str(e))
            import traceback
            traceback.print_exc()

    @property
    def running(self):
        return self._running

    def _runFinished(self):
        self._running = False
        try:
            self._asynchrony.GetOutput()
            if self._finishCallback is not None:
                self._finishCallback(None)
        except Asynchrony.CancelledException:
            # if they cancelled, the finish was as expected
            if self._finishCallback is not None:
                self._finishCallback(None)
        except Exception as e:
            if self._finishCallback is not None:
                self._finishCallback(e)
        finally:
            self._asynchrony = None

    def setFinishedCallback(self, finishCallback=None):
        self._finishCallback = finishCallback

    def startTBLog(self):
        """
        Starts tensorboard logging
        """
        tb_dirs = os.path.join(self.ui.writeDirLineEdit.text, "logs", self.model)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tb_dirs, '--port', self.ui.tbPortLineEdit.text])
        try:
            tb.launch()
            self.tb_log = tb
            return tb
        except tensorboard.program.TensorBoardPortInUseError:
            msg = qt.QMessageBox()
            msg.setIcon(qt.QMessageBox.Information)
            msg.setText("Port in use by a previous session "
                        "which will be displayed!")
            msg.exec()
            logging.info("Port in use by a previous session "
                         "which will be displayed!")
            return self.tb_log

    def showTBLog(self):
        """
        Start tensorboard log web ui
        """
        Asynchrony(self.startTBLog())
        if not "http" in self.ui.tbAddressLineEdit.text:
            tb_url = "http://"
        else:
            tb_url = ""
        tb_url += self.ui.tbAddressLineEdit.text + ":" + self.ui.tbPortLineEdit.text
        logging.info(f"Opening log in a browser: {tb_url}")
        webbrowser.open(tb_url)


#
# DeepLearnerLogic
#

class DeepLearnerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
            Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
            Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(
            self,
            in_channels,
            num_classes,
            model_name,
            batch_size,
            learning_rate,
            max_epochs,
            n_folds,
            use_gpu,
            logdir,
            exp_name="tensorboard",
            cp_n_epoch=1,
            max_cp=-1,
            monitor="validation/valid_loss",
            pos_weight=1.0,
            file_paths=None,
            ui=None,
            w=None
        ):
        """
        Run the processing algorithm.
        Can be used without GUI widget.

        :param int in_channels: Number of input channels in the images (For example, it should be 3 for RGB images)
        :param int num_classes: Number of classes that exist in the dataset
        :param str model_name: The string for the model which will be trained
        :param int batch_size: Number of training samples in each batch while training
        :param float learning_rate: learning rate for the training
        :param int max_epochs: Maximum number of epochs the model will be trained.
        :param int n_folds: Number of cross-validation folds
        :param bool use_gpu: whether to train on gpu or cpu (False for cpu and True for gpu)
        :param str logdir: Directory where all the outputs will be saved (like checkpoints and tensorboard logs)
        :param str exp_name: Name of the experiment. A new folder named `exp_name` will be created inside the `logdir`
        for each experiment.
        :param int cp_n_epoch: A new checkpoint will be saved every `cp_n_epoch`
        :param max_cp: The best `max_cp` checkpoints will be kept (Use -1 to keep them all).
        :param monitor: The value based on which the quality of checkpoint will be assessed. Possible values
        :param pos_weight: The weight of the positive class in the loss function
        "train/train_loss", "validation/valid_loss", "train/acc", "validation/acc", "train/precision",
        "validation/precision", "train/recall", "validation/recall"
        :param file_paths: A dictionary containing relevant paths to training data. (Default: FILE_PATH object in CONSTANTS.py)
        :param ui: The UI object
        :param w: image size
        """
        args = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "max_epochs": max_epochs,
            "gpus": use_gpu,
            "model": model_name,
            "n_folds": n_folds,
            "data_workers": 1,
            "write_dir": logdir,
            "exp_name": exp_name,
            "cp_n_epoch": cp_n_epoch,
            "maxCp": max_cp,
            "monitor": monitor,
            "pos_weight": pos_weight,
            "qtProgressBarObject": ui.trainingProgressBar,
            "file_paths": file_paths,
            "w": w
        }
        import time
        from DeepLearnerLib.training.EfficientNetTrainer import cli_main
        startTime = time.time()
        logging.info('Processing started ... ')
        logging.info(args)
        cli_main(args)
        ui.trainingProgressBar.setValue(100)
        ui.StartTrain.enabled = True
        self.InputDirPushButton.enabled = True
        stopTime = time.time()
        logging.info('Processing completed in {0:.2f} seconds'.format(stopTime - startTime))


#
# DeepLearnerTest
#

class DeepLearnerTest(ScriptedLoadableModuleTest):
    """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
    """
        self.setUp()
        self.test_DeepLearner1()

    def test_DeepLearner1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

        self.delayDisplay("Starting the test")

        # Get/create input data

        # inputVolume = SampleData.downloadSample('DeepLearner1')
        # self.delayDisplay('Loaded test data set')
        #
        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)
        #
        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100
        #
        # # Test the module logic
        #
        # logic = DeepLearnerLogic()
        #
        # # Test algorithm with non-inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, True)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)
        #
        # # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
