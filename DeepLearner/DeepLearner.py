import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from pathlib import Path


#
# DeepLearner
#
from src.training.EfficientNetTrainer import cli_main


class DeepLearner(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DeepLearner"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Deep Learning"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Md Asadullah Turja (UNC Chapel Hill)"]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """"""
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
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/DeepLearner.ui'))
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
        self.ui.maxEpochLineEdit.text = "10"
        self.ui.batchSizeLineEdit.text = "2"
        self.ui.learningRateLineEdit.text = "1e-3"
        self.ui.writeDirLineEdit.text = os.path.join(Path.home(), "defaultExperiment")
        self.ui.tbPortLineEdit.text = "6010"
        self.ui.tbAddressLineEdit.text = "localhost"
        self.ui.nEpoch.text = "5"
        self.ui.maxCpLineEdit.text = "2"
        self.ui.monitorLineEdit.text = "validation/valid_loss"

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.TrainDirLineEdit.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.maxEpochLineEdit.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.gPUCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.batchSizeLineEdit.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.learningRateLineEdit.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

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
        return 0

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

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        try:
            # Compute output
            args = {
                "batch_size": int(self.ui.batchSizeLineEdit.text),
                "learning_rate": float(self.ui.learningRateLineEdit.text),
                "in_channels": 2,
                "num_classes": 2,
                "max_epochs": int(self.ui.maxEpochLineEdit.text),
                "gpus": 0,
                "model": self.model,
                "logdir": os.path.join(self.ui.writeDirLineEdit.text, "tb_logs"),
                "n_folds": 2,
                "data_workers": 1,
                "write_dir": self.ui.writeDirLineEdit.text,
                "exp_name": "default",
                "cp_n_epoch": int(self.ui.nEpoch.text),
                "maxCp": int(self.ui.maxCpLineEdit.text),
                "monitor": self.ui.monitorLineEdit.text
            }
            self.logic.process(args)

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

    def process(self, args):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        """

        import time
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
