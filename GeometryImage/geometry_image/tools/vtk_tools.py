import vtk
import numpy as np


def ReadPolyData(file_name):
    import os
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    poly_data = vtk.vtkPolyData()
    if extension == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        # poly_data = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        # poly_data = reader.GetOutput()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        # poly_data = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        # poly_data = reader.GetOutput()
    elif extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        # poly_data = reader.GetOutput()
    elif extension == ".g":
        reader = vtk.vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        # poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        return None
    poly_data.ShallowCopy(reader.GetOutput())
    return poly_data


def renderPolyData(poly_data):
    ren = vtk.vtkRenderer()
    renWindow = vtk.vtkRenderWindow()
    renWindow.AddRenderer(ren)
    renWindow.SetWindowName('Model')
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWindow)

    actor = vtk.vtkActor()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    actor.SetMapper(mapper)
    ren.AddActor(actor)
    # renWindow.SetSize(512, 512)
    renWindow.Render()
    iren.Start()


def updateVerticesInVTK(poly_data, vertices):
    """
    Creates a new poly data by replacing
     the value of the vertices of a poly data object without changing the id.
    :param poly_data: original poly data object
    :param vertices: numpy array of points
    :return: new poly data object after replacement
    """

    new_poly_data = vtk.vtkPolyData()
    new_poly_data.DeepCopy(poly_data)

    # Add new points
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(poly_data.GetNumberOfPoints())
    for id in range(poly_data.GetNumberOfPoints()):
        points.SetPoint(id, tuple(vertices[id]))
    new_poly_data.SetPoints(points)

    # Copy Triangles
    # cell = vtk.vtkCellArray()
    # for id in range(poly_data.GetNumberOfPolys()):
    #     cell.InsertNextCell(poly_data.GetCell(id))
    # new_poly_data.SetPolys(cell)

    # Copy Point Data
    # point_data = vtk.vtkDoubleArray()
    # for id in range(poly_data.GetNumberOfPoints()):
    #     point_data.InsertNextTuple(poly_data.GetPointData(0))
    # new_poly_data.GetPointData().GetArray()

    return new_poly_data


if __name__ == '__main__':
    from vtk.util.numpy_support import vtk_to_numpy
    poly_data = ReadPolyData("../Sphere_Template/sphere_f20480_v10242.vtk")
    print(poly_data)
    numpy_points = vtk_to_numpy(poly_data.GetPointData().GetArray(0))
    new_poly_data = updateVerticesInVTK(poly_data, numpy_points)
    renderPolyData(new_poly_data)


