import os
import vtk


# calculate total size of all files inside a folder
def folder_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


# label for a viewport
# using normalized viewport coordinates
def make_label(text: str, x=0.02, y=0.95, font_size=20):
    actor = vtk.vtkTextActor()
    actor.SetInput(text)

    prop = actor.GetTextProperty()
    prop.SetFontSize(font_size)
    prop.SetColor(1.0, 1.0, 1.0)
    prop.SetBold(True)

    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(x, y)
    return actor


# folder that contains DICOM slices
dicom_dir = "Skull_Dataset" # Source: https://www.kaggle.com/datasets/trainingdatapro/fazekas-mri

# check if folder exists
if not os.path.isdir(dicom_dir):
    raise FileNotFoundError(
        f"Cannot find DICOM directory: '{dicom_dir}'. "
        f"Put your DICOM slices inside this folder or update dicom_dir."
    )

# read DICOM series
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(dicom_dir)
reader.Update()

# get image data object
image = reader.GetOutput()

# get dataset info for report
min_intensity, max_intensity = image.GetScalarRange()
dims = image.GetDimensions()
spacing = image.GetSpacing()   # voxel size
origin = image.GetOrigin()
size_bytes = folder_size_bytes(dicom_dir)
size_mb = size_bytes / (1024 * 1024)
vtk_version = vtk.vtkVersion.GetVTKVersion()

# print dataset info
print("Dataset Info of Cancer Cell of Brain")
print(f"Dimensions (Nx, Ny, Nz): {dims}")
print(f"Spacing / Voxel size (sx, sy, sz): {spacing}")
print(f"Origin: {origin}")
print(f"Scalar range (min, max): ({min_intensity}, {max_intensity})")
print(f"DICOM folder size: {size_mb:.2f} MB")
print(f"VTK Version: {vtk_version}")

# opacity transfer function for volume rendering
# low values stay transparent, higher values become visible
opacity = vtk.vtkPiecewiseFunction()
opacity.AddPoint(-2048, 0.00)
opacity.AddPoint(-1000, 0.00)
opacity.AddPoint(-300,  0.00)
opacity.AddPoint(0,     0.03)
opacity.AddPoint(200,   0.08)
opacity.AddPoint(500,   0.18)
opacity.AddPoint(900,   0.35)
opacity.AddPoint(1400,  0.60)
opacity.AddPoint(5449,  0.85)

# color transfer function for volume rendering
color = vtk.vtkColorTransferFunction()
color.AddRGBPoint(-2048, 0.0, 0.0, 0.0)
color.AddRGBPoint(-500,  0.0, 0.0, 0.0)
color.AddRGBPoint(0,     0.35, 0.15, 0.12)
color.AddRGBPoint(300,   0.75, 0.35, 0.20)
color.AddRGBPoint(800,   0.90, 0.75, 0.45)
color.AddRGBPoint(1400,  0.95, 0.90, 0.80)
color.AddRGBPoint(5449,  1.00, 1.00, 0.95)

# volume rendering properties
volume_prop = vtk.vtkVolumeProperty()
volume_prop.SetColor(color)
volume_prop.SetScalarOpacity(opacity)
volume_prop.ShadeOn()
volume_prop.SetInterpolationTypeToLinear()
volume_prop.SetAmbient(0.20)
volume_prop.SetDiffuse(0.90)
volume_prop.SetSpecular(0.20)
volume_prop.SetSpecularPower(10.0)

# volume mapper
volume_mapper = vtk.vtkSmartVolumeMapper()
volume_mapper.SetInputConnection(reader.GetOutputPort())

# volume actor
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_prop)

# iso-surface extraction using marching cubes algorithm
iso_value = 310

mc = vtk.vtkMarchingCubes()
mc.SetInputConnection(reader.GetOutputPort())
mc.ComputeGradientsOn()
mc.ComputeNormalsOff()
mc.SetValue(0, iso_value)

# compute normals to improve iso-surface shading
normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(mc.GetOutputPort())
normals.SetFeatureAngle(60.0)
normals.ConsistencyOn()
normals.SplittingOff()

# mapper for iso-surface
iso_mapper = vtk.vtkPolyDataMapper()
iso_mapper.SetInputConnection(normals.GetOutputPort())
iso_mapper.ScalarVisibilityOff()

# iso-surface actor
iso_actor = vtk.vtkActor()
iso_actor.SetMapper(iso_mapper)
iso_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
iso_actor.GetProperty().SetSpecular(0.2)
iso_actor.GetProperty().SetSpecularPower(20.0)

# create 3 renderers (3 viewports)
volumeRenderer = vtk.vtkRenderer()
isoRenderer = vtk.vtkRenderer()
comboRenderer = vtk.vtkRenderer()

# set viewport positions
volumeRenderer.SetViewport(0.00,   0.0, 0.3333, 1.0)
isoRenderer.SetViewport(0.3333,    0.0, 0.6666, 1.0)
comboRenderer.SetViewport(0.6666,  0.0, 1.0000, 1.0)

# viewport 1: volume only
volumeRenderer.AddVolume(volume)

# viewport 2: iso-surface only
isoRenderer.AddActor(iso_actor)

# viewport 3: both volume and iso-surface
comboRenderer.AddVolume(volume)
comboRenderer.AddActor(iso_actor)

# set background colors
volumeRenderer.SetBackground(0.10, 0.10, 0.12)
isoRenderer.SetBackground(0.08, 0.08, 0.10)
comboRenderer.SetBackground(0.10, 0.08, 0.10)

# add labels to all viewports
volumeRenderer.AddActor2D(make_label("Viewport 1: Volume Rendering"))
isoRenderer.AddActor2D(make_label(f"Viewport 2: Iso-surface (MC, value={iso_value})"))
comboRenderer.AddActor2D(make_label("Viewport 3: Volume + Iso-surface"))

# dataset info in viewport 3
info_text = (
    f"Dims: {dims[0]}x{dims[1]}x{dims[2]}\n"
    f"Voxel: {spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}\n"
    f"Range: {min_intensity:.0f} to {max_intensity:.0f}"
)
info_actor = vtk.vtkTextActor()
info_actor.SetInput(info_text)
info_prop = info_actor.GetTextProperty()
info_prop.SetFontSize(14)
info_prop.SetColor(0.9, 0.9, 0.9)
info_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
info_actor.SetPosition(0.02, 0.06)
comboRenderer.AddActor2D(info_actor)

# shared camera on all 3 viewports
shared_camera = vtk.vtkCamera()
volumeRenderer.SetActiveCamera(shared_camera)
isoRenderer.SetActiveCamera(shared_camera)
comboRenderer.SetActiveCamera(shared_camera)

# reset camera and zoom
volumeRenderer.ResetCamera()
shared_camera.Zoom(1.2)

# create render window
renWin = vtk.vtkRenderWindow()
renWin.SetSize(1800, 800)
renWin.AddRenderer(volumeRenderer)
renWin.AddRenderer(isoRenderer)
renWin.AddRenderer(comboRenderer)

# create interactor for mouse controls
winInteract = vtk.vtkRenderWindowInteractor()
winInteract.SetRenderWindow(renWin)
style = vtk.vtkInteractorStyleTrackballCamera()
winInteract.SetInteractorStyle(style)

# render and start interaction
renWin.Render()
winInteract.Initialize()
winInteract.Start()
