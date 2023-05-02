import os
import pathlib
import unittest
import logging

try:
    import numpy as np
except ImportError:
    slicer.util.pip_install('numpy==1.21.2')
    import numpy as np

try:
    import torch
except ImportError:
    slicer.util.pip_install('torch==1.9.0')
    import torch

try:
    import pytorch_lightning as pl
except ImportError:
    slicer.util.pip_install('pytorch_lightning==1.4.9')
    import pytorch_lightning as pl

try:
    import sklearn
except ImportError:
    slicer.util.pip_install('scikit-learn==0.24.2')

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from InferenceLib.CONSTANTS import DEFAULT_FILE_PATHS
from InferenceLib.Asynchrony import Asynchrony

#
# Inference
#
from InferenceLib.data_utils.GeomCnnDataset import GeomCnnDataModule


class Inference(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Inference"
        self.parent.categories = [
            "Deep Learning"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Md Asadullah Turja (UNC Chapel Hill)"]
        self.parent.helpText = """
        This is the inference module that accompanies the training module (DeepLearnerLib) 
        for applying the trained models on new datasets. The dataset must have the same hierarchy 
        as the training dataset. The model path can be either *.pt file or the folder containing 
        the output of all the folds from the training module.
        Github link: 
        """
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """dummy"""


#

class InferenceWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self._finishCallback = None
        self._running = None
        self._asynchrony = None
        self.modelDir = None
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.testDir = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/Inference.ui'))
        self.uiWidget = uiWidget
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Default values
        self.ui.inferenceProgressBar.setValue(0)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = InferenceLogic(self.ui.inferenceProgressBar)

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.predictPushButton.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.TestDirPushButton.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.ModelDirPushButton.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.predictPushButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.TestDirPushButton.connect('clicked(bool)', self.populateTestDirectory)
        self.ui.ModelDirPushButton.connect('clicked(bool)', self.populateModelDirectory)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def populateDataDirectory(self):
        DEFAULT_FILE_PATHS["TEST_DATA_DIR"] = self.ui.TestDirLineEdit.text
        DEFAULT_FILE_PATHS["TRAIN_DATA_DIR"] = self.ui.TestDirLineEdit.text
        DEFAULT_FILE_PATHS["FEATURE_DIRS"] = [self.ui.FeatureNameCombo.currentText]
        DEFAULT_FILE_PATHS["TIME_POINTS"] = [self.ui.TimePointCombo.currentText]
        return DEFAULT_FILE_PATHS

    def populateInputDirectory(self, type="model"):
        """
        Populates self.inputPath
        """
        dialog = qt.QFileDialog(self.uiWidget)
        if type == "data":
            dialog.setFileMode(qt.QFileDialog.Directory)
        dialog.setViewMode(qt.QFileDialog.Detail)
        if dialog.exec_():
            directoryPath = dialog.selectedFiles()
            logging.debug("onDirectoryChanged: {}".format(directoryPath))
            if type == "data":
                self.testDir = directoryPath[0]
                self.ui.TestDirLineEdit.text = directoryPath[0]
                feat_set = set()
                timepoints = set()
                for sub in os.listdir(self.testDir):
                    if not os.path.isdir(os.path.join(self.testDir, sub)):
                        continue
                    for tp in os.listdir(os.path.join(self.testDir, sub)):
                        if not os.path.isdir(os.path.join(self.testDir, sub, tp)):
                            continue
                        timepoints.add(tp)
                        for feat in os.listdir(os.path.join(self.testDir, sub, tp)):
                            if not os.path.isdir(os.path.join(self.testDir, sub, tp, feat)):
                                continue
                            feat_set.add(feat)
                for f in feat_set:
                    self.ui.FeatureNameCombo.addItem(f)
                for tp in timepoints:
                    self.ui.TimePointCombo.addItem(tp)
            elif type == "model":
                self.modelDir = directoryPath[0]
                self.ui.ModelDirLineEdit.text = directoryPath[0]

    def populateTestDirectory(self):
        self.populateInputDirectory("data")

    def populateModelDirectory(self):
        self.populateInputDirectory("model")

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

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        pass

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        pass

    def constructWriteDirectory(self):
        baseWriteDir = self.ui.OutputDirectoryLineEdit.text
        print(f"Write dir: {baseWriteDir}")
        if baseWriteDir.strip() == "" or baseWriteDir is None:
            return None
        else:
            save_path = os.path.join(
                baseWriteDir,
                self.ui.TimePointCombo.currentText,
                self.ui.FeatureNameCombo.currentText
            )
            pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
            return save_path

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        try:
            file_paths = self.populateDataDirectory()
            save_path = self.constructWriteDirectory()
            if save_path is None:
                raise Exception("Output Directory can't be empty")
            # if os.path.isdir(self.ui.ModelDirLineEdit.text):
            #     save_path = self.ui.ModelDirLineEdit.text
            # else:
            #     save_path = os.path.dirname(self.ui.ModelDirLineEdit.text)
            self._asynchrony = Asynchrony(
                lambda: self.logic.process(
                    file_paths=file_paths,
                    model_path=self.ui.ModelDirLineEdit.text,
                    save_path=save_path)
            )
            self._asynchrony.Start()
            self._running = True
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


#
# InferenceLogic
#

def setProgressBar(qtProgressBarObject, percent):
    qtProgressBarObject.setValue(percent)


class InferenceLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(
            self,
            progressBar):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.n_fold = 1
        self.progressBar = progressBar

    def predict(
            self,
            model,
            data_loader,
            fold_id=0
    ):
        preds = []
        count = 1
        sample_size = len(data_loader)
        for x, y in data_loader:
            y_hat = model(x)
            y_hat = torch.argmax(torch.nn.Softmax(dim=-1)(y_hat), dim=-1)
            preds.append(y_hat)
            val = ((count + sample_size * fold_id) * 100) // (sample_size * self.n_fold)
            Asynchrony.RunOnMainThread(lambda: self.progressBar.setValue(val))
            count += 1
        preds = torch.cat(preds)
        return preds

    def process(
            self,
            file_paths,
            model_path,
            save_path
    ):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param data_path: volume to be thresholded
        :param model_path: thresholding result
        :param save_path: values above/below this threshold will be set to 0
        """

        import time
        startTime = time.time()
        logging.info('Processing started ... ')
        data_loader = GeomCnnDataModule(
            batch_size=16,
            num_workers=4,
            file_paths=file_paths
        ).test_dataloader()

        # Load model
        # There are two possible options:
        #   1. The path can just be a file in which case the model from the
        # file is loaded for prediction.
        #   2. The path can be the "Output Directory/logs/<model_name>"
        # from the training module where the outputs from all the folds are present.
        # In this case a prediction based on the majority voting will be computed
        # using the models from the folds in that directory.

        if os.path.isfile(model_path):
            self.n_fold = 1
            model = torch.load(model_path)
            predictions = self.predict(model, data_loader, 0)
        else:
            # majority voting
            predictions = []
            fold_model_paths = [os.path.join(model_path, fold, "model.pt")
                                for fold in os.listdir(model_path)
                                if os.path.isdir(os.path.join(model_path, fold))]
            self.n_fold = len(fold_model_paths)
            for i, path in enumerate(fold_model_paths):
                model = torch.load(path)
                preds = self.predict(model, data_loader, fold_id=i)
                np.savetxt(os.path.join(os.path.dirname(path),
                                        "predictions.csv"),
                           preds.unsqueeze(-1).cpu().numpy(), delimiter=",")
                predictions.append(preds)
            predictions = torch.stack(predictions, dim=-1)
            predictions = torch.mode(predictions, dim=-1, keepdim=True)[0]

        np.savetxt(os.path.join(save_path, "predictions.csv"),
                   predictions.cpu().numpy(), delimiter=",")
        stopTime = time.time()
        logging.info('Processing completed in {0:.2f} seconds'.format(stopTime - startTime))
        Asynchrony.RunOnMainThread(
            lambda: slicer.util.infoDisplay(f"Prediction results saved in {save_path}/predictions.csv"))
