import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from pathlib import Path
from DeepLearnerLib.CONSTANTS import DEFAULT_FILE_PATHS

#
# DeepLearnerLib
#


class DeepLearner(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DeepLearnerLib"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Deep Learning"]  # TODO: set categories (folders where the module shows up in the module selector)
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
        self.logic = None
        self.model = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/DeepLearnerLib.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.uiWidget = uiWidget
        self.logic = DeepLearnerLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Initialize training hyperparameters with default values
        # self.ui.FeatureNameLineEdit.text = ""
        # self.ui.TimePointLineEdit.text = ""
        self.ui.CVFoldLineEdit.text = "5"
        # self.ui.epochSpinBox.text = ""
        # self.ui.batchSizeLineEdit.text = ""
        # self.ui.learningRateSpinBox.text = ""
        self.ui.writeDirLineEdit.text = os.path.join(Path.home(), "defaultExperiment")
        self.ui.tbPortLineEdit.text = "6010"
        self.ui.tbAddressLineEdit.text = "localhost"
        self.ui.monitorLineEdit.text = "validation/valid_loss"

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.TrainDirLineEdit.connect("textChanged(str)", self.updateParameterNodeFromGUI)
        self.ui.epochSpinBox.connect("textChanged(str)", self.updateParameterNodeFromGUI)
        self.ui.gPUCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.batchSizeSpinBox.connect("textChanged(str)", self.updateParameterNodeFromGUI)
        self.ui.learningRateSpinBox.connect("textChanged(str)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.TrainDirPushButton.connect('clicked(bool)', self.populateTrainDirectory)
        self.ui.StartTrain.connect('clicked(bool)', self.onApplyButton)
        self.ui.showLogPushButton.connect('clicked(bool)', self.showTBLog)
        # Model radio buttons
        self.ui.DenseNetRadio.toggled.connect(self.processRadioButton)
        self.ui.EfficientNetRadio.toggled.connect(self.processRadioButton)
        self.ui.ResNetRadio.toggled.connect(self.processRadioButton)
        self.ui.SimpleCNNRadio.toggled.connect(self.processRadioButton)
        self.ui.SimpleCNNRadio.checked = True


        # Buttons
        # self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

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
        # self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        # self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        # self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

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
                self.ui.TrainDirLineEdit.text = directoryPath[0]
                feat_set = set()
                timepoints = set()
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
                for f in feat_set:
                    self.ui.FeatureNameCombo.addItem(f)
                for tp in timepoints:
                    self.ui.TimePointCombo.addItem(tp)

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

    def populateDataDirectory(self):
        DEFAULT_FILE_PATHS["TRAIN_DATA_DIR"] = self.ui.TrainDirLineEdit.text
        DEFAULT_FILE_PATHS["FEATURE_DIRS"] = [self.ui.FeatureNameCombo.currentText]
        DEFAULT_FILE_PATHS["TIME_POINTS"] = [self.ui.TimePointCombo.currentText]
        return DEFAULT_FILE_PATHS

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        try:
            # Compute output
            file_paths = self.populateDataDirectory()
            print("File path: ", file_paths)

            self.logic.process(
                in_channels=2,
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
                file_paths=file_paths
            )
            self.ui.trainingProgressBar.setValue(100)

        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: " + str(e))
            import traceback
            traceback.print_exc()

    def startTBLog(self):
        """
        Starts tensorboard logging
        """
        import subprocess
        from multiprocessing import Process
        cmd = ["tensorboard", "--logdir", self.ui.logDirectoryLineEdit.text, "--port", self.ui.tbPortLineEdit.text]
        # tb_tool = TensorBoardTool(self.ui.logDirectoryLineEdit.text, self.ui.tbPortLineEdit.text)
        # p = Process(target=tb_tool.run)
        # p.start()
        print(f"Running subprocess: {cmd}")
        subprocess.Popen(cmd)

    def showTBLog(self):
        """
      Start tensorboard log web ui
      """
        # self.startTBLog()
        if not "http" in self.ui.tbAddressLineEdit.text:
            tb_url = "http://"
        else:
            tb_url = ""
        tb_url += self.ui.tbAddressLineEdit.text + ":" + self.ui.tbPortLineEdit.text
        print("Opening tb log: ", tb_url)
        self.webWidget = slicer.qSlicerWebWidget()
        self.webWidget.url = qt.QUrl(tb_url)
        self.webWidget.show()


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
            exp_name="default",
            cp_n_epoch=1,
            max_cp=-1,
            monitor="validation/valid_loss",
            progressbar=None,
            file_paths=None
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
        "train/train_loss", "validation/valid_loss", "train/acc", "validation/acc", "train/precision",
        "validatin/precision", "train/recall", "validation/recall"
        :param progressbar: A progress bar object that will be updated during training
        :param file_paths: A dictionary containing relevant paths to training data. (Default: FILE_PATH object in CONSTANTS.py)
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
            "qtProgressBarObject": progressbar,
            "file_paths": file_paths
        }
        import time
        from DeepLearnerLib.training.EfficientNetTrainer import cli_main
        startTime = time.time()
        logging.info('Processing started ... ')
        logging.info(args)
        cli_main(args)
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
