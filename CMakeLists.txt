cmake_minimum_required(VERSION 3.13.4)

project(DeepLearningWizard)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/DeepLearningUI")
set(EXTENSION_CATEGORY "Deep Learning")
set(EXTENSION_CONTRIBUTORS "Md Asadullah Turja (University of North Carolina at Chapel Hill)")
set(EXTENSION_DESCRIPTION "This extention creates a UI for training deep learning models such as EfficientNet, ResNet etc.  It is developed for training medical images such Brain MRI data.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/DeepLearningUI.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/DeepLearningUI/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DeepLearner)
add_subdirectory(Deep)
add_subdirectory(Inference)
add_subdirectory(GeometryImage)
add_subdirectory(GeometryImage)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
