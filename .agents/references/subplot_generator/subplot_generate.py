
from typing import Optional
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QMessageBox
from PySide6.QtCore import Signal, Slot
from qfluentwidgets import (
    PushButton,
    PrimaryPushButton,
    ComboBox,
    SpinBox,
    DoubleSpinBox,
    CheckBox,
    InfoBar,
    PushButton,
    PrimaryPushButton,
    ComboBox,
    SpinBox,
    DoubleSpinBox,
    CheckBox,
    InfoBar,
    BodyLabel,
    StrongBodyLabel,
    SubtitleLabel
)

from loguru import logger
from pathlib import Path

from loguru import logger
from pathlib import Path

from src.gui.components.base_interface import TabInterface, PageGroup
from src.gui.config import tr
from src.core.subplot_generator import SubplotGenerator


class SubplotTab(TabInterface):
    """
    Interface content for Subplot Generation.
    """

    sigLoadImage = Signal()
    sigLoadBoundary = Signal()
    sigPreview = Signal()
    sigGenerate = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_layout(self):
        super()._init_layout()
        
        # Connect Map Canvas signals
        self.map_component.map_canvas.sigRotationChanged.connect(self._on_canvas_rotation_changed)

    def _on_canvas_rotation_changed(self, angle):
        """Handle rotation change from map canvas."""
        subplot_panel = self.property_panel.get_subplot_panel()
        subplot_panel.spin_rotation.blockSignals(True)
        subplot_panel.spin_rotation.setValue(angle)
        subplot_panel.spin_rotation.blockSignals(False)

    def _init_ui(self) -> None:
        """Initialize the controls for subplot generation."""
        # --- File Group ---
        file_group = PageGroup(tr("page.subplot.group.file"))

        self.btn_load_image = PushButton(tr("page.subplot.btn.load_image"))
        self.btn_load_image.clicked.connect(self._on_load_image)
        file_group.add_widget(self.btn_load_image)

        self.btn_load_boundary = PushButton(tr("page.subplot.btn.load_boundary"))
        self.btn_load_boundary.clicked.connect(self._on_load_boundary)
        file_group.add_widget(self.btn_load_boundary)

        self.add_group(file_group)

        # --- View Group ---
        view_group = PageGroup(tr("page.subplot.group.view")) # Reusing action key for View

        self.check_preview = CheckBox(tr("page.subplot.check.preview"))
        self.check_preview.setChecked(True)
        self.check_preview.stateChanged.connect(self._auto_preview)
        view_group.add_widget(self.check_preview)

        self.btn_focus = PushButton(tr("page.subplot.btn.focus"))
        self.btn_focus.clicked.connect(self._on_focus)
        view_group.add_widget(self.btn_focus)

        self.add_group(view_group)
        self.add_stretch()
        
        # Connect Action Signals from Property Panel
        panel = self.property_panel.get_subplot_panel()
        panel.sigGenerate.connect(self._on_generate)
        panel.sigReset.connect(self._on_reset)
        panel.sigParamChanged.connect(self._auto_preview)

        # Init Generator
        self.generator = SubplotGenerator()
        self.boundary_gdf = None

        # Init Maps
        # self.map_component comes from MapInterface

    @Slot()
    def _on_load_image(self):
        """Load background image (DOM)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr("page.subplot.dialog.load_image"), "", "Image Files (*.tif *.tiff *.png *.jpg);;All Files (*)"
        )
        if file_path:
            logger.info(f"User selected image: {file_path}")
            if self.map_component.map_canvas.add_raster_layer(file_path):
                logger.info("Image loaded successfully.")
                # Show success message
                InfoBar.success(
                    title=tr("success"),
                    content=f"Loaded: {Path(file_path).name}",
                    parent=self,
                    duration=3000
                )
                
                # Zoom to image if no boundary loaded
                if self.boundary_gdf is None:
                    self.map_component.map_canvas.zoom_to_layer(Path(file_path).stem)
                    
            else:
                logger.error(f"Failed to load image: {file_path}")
                InfoBar.error(
                    title=tr("error"),
                    content=f"Failed to load image: {file_path}",
                    parent=self
                )

    @Slot()
    def _on_load_boundary(self):
        """Load boundary SHP."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr("page.subplot.dialog.load_boundary"), "", "Shapefile (*.shp);;All Files (*)"
        )
        if file_path:
            gdf = self.generator.load_boundary(file_path)
            if gdf is not None:
                self.boundary_gdf = gdf
                self.map_component.map_canvas.add_vector_layer(
                    gdf, "Boundary", color='#FF0000', width=2
                )
                self._auto_preview()
            else:
                InfoBar.error(
                    title=tr("error"),
                    content=tr("page.subplot.error.invalid_boundary"),
                    parent=self
                )

    @Slot()
    def _on_mode_changed(self, index: int):
        """Handle definition mode change."""
        pass # Now handled in Property Panel

    @Slot()
    def _auto_preview(self):
        """Generate preview if checkbox is checked."""
        if not self.check_preview.isChecked() or self.boundary_gdf is None:
            return
        
        try:
            panel = self.property_panel.get_subplot_panel()
            mode = panel.combo_def_mode.currentIndex() # 0: RC, 1: Size
            x_space = panel.spin_x_spacing.value()
            y_space = panel.spin_y_spacing.value()

            if mode == 0:
                rows = panel.spin_rows.value()
                cols = panel.spin_cols.value()
                # Call generator logic for RC mode
                gdf = self.generator.generate(
                    self.boundary_gdf, rows, cols, x_space, y_space, output_path=None
                )
            else:
                 # Calculate Rows/Cols from Size (Cell Width/Height)
                 cell_width = panel.spin_width.value()
                 cell_height = panel.spin_height.value()
                 
                 bounds = self.boundary_gdf.total_bounds # minx, miny, maxx, maxy
                 total_width = bounds[2] - bounds[0]
                 total_height = bounds[3] - bounds[1]
                 
                 # Width = Cols * CellWidth + (Cols - 1) * XMethod
                 # TotalWidth approx Cols * (CellWidth + XSpace) - XSpace
                 # Cols = (TotalWidth + XSpace) / (CellWidth + XSpace)
                 
                 if cell_width + x_space > 0:
                    cols = int(round((total_width + x_space) / (cell_width + x_space)))
                 else:
                    cols = 1
                    
                 if cell_height + y_space > 0:
                    rows = int(round((total_height + y_space) / (cell_height + y_space)))
                 else:
                    rows = 1
                 
                 cols = max(1, cols)
                 rows = max(1, rows)

                 gdf = self.generator.generate(
                    self.boundary_gdf, rows, cols, x_space, y_space, output_path=None
                )
            
            if gdf is not None:
                self.map_component.map_canvas.add_vector_layer(
                    gdf, "Preview", color='#00FF00', width=1
                )
        except Exception as e:
            # Don't show popup on auto preview error, just log
            print(f"Preview error: {e}")

    @Slot()
    def _on_focus(self):
        """Focus on the boundary layer."""
        if self.boundary_gdf is not None:
            # Focus logic here (usually handled by map_component)
            self.map_component.map_canvas.zoom_to_layer("Boundary")

            # Auto-rotate
            angle = self.generator.calculate_optimal_rotation(self.boundary_gdf)
            if angle is not None:
                self.map_component.map_canvas.set_rotation(angle)
            else:
                self.map_component.map_canvas.set_rotation(0)

    @Slot()
    def _on_generate(self):
        """Generate subplots and save."""
        if self.boundary_gdf is None:
            InfoBar.warning(
                title=tr("warning"),
                content=tr("page.subplot.msg.no_boundary"),
                parent=self
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, tr("page.subplot.dialog.save"), "", "Shapefile (*.shp)"
        )

        if not file_path:
            return

        try:
            panel = self.property_panel.get_subplot_panel()
            mode = panel.combo_def_mode.currentIndex()
            x_space = panel.spin_x_spacing.value()
            y_space = panel.spin_y_spacing.value()

            if mode == 0:
                rows = panel.spin_rows.value()
                cols = panel.spin_cols.value()
            else:
                 # Recalculate for final generation to be safe
                 cell_width = panel.spin_width.value()
                 cell_height = panel.spin_height.value()
                 
                 bounds = self.boundary_gdf.total_bounds
                 total_width = bounds[2] - bounds[0]
                 total_height = bounds[3] - bounds[1]
                 
                 if cell_width + x_space > 0:
                    cols = int(round((total_width + x_space) / (cell_width + x_space)))
                 else:
                    cols = 1
                    
                 if cell_height + y_space > 0:
                    rows = int(round((total_height + y_space) / (cell_height + y_space)))
                 else:
                    rows = 1
                 
                 cols = max(1, cols)
                 rows = max(1, rows)

            self.generator.generate(
                self.boundary_gdf, rows, cols, x_space, y_space, output_path=file_path
            )
            
            InfoBar.success(
                title=tr("success"),
                content=tr("page.subplot.msg.success"),
                parent=self
            )
            
            # Load the generated result
            # self.map_component.map_canvas.add_vector_layer(
            #     file_path, "Result", color='#0000FF', width=2
            # )
            
        except Exception as e:
            InfoBar.error(
                title=tr("error"),
                content=f"Generation failed: {e}",
                parent=self
            )

    @Slot()
    def _on_reset(self):
        """Reset parameters."""
        # TODO: Implement reset logic if needed
        pass
