# These methods need to be added after superres_step1_detect_bf in mib_viewer_pyqtgraph.py
# Insert right before the auto_detect_and_load_fft method (around line 4365)

def superres_step2_correlations(self):
    """Step 2: Compute cross-correlations"""
    if self.superres_data_cropped is None:
        return

    try:
        from ..processing.superres_processor import SuperResProcessor

        self.sr_step2_btn.setEnabled(False)
        self.sr_step2_status.setText("Computing cross-correlations (~20s)...")
        QApplication.processEvents()

        processor = SuperResProcessor()
        ref_smoothing = self.sr_ref_smoothing.value()

        # Compute correlations
        correlations, reference_image = processor.compute_cross_correlations(
            self.superres_data_cropped, ref_smoothing
        )

        # Cache results
        self.superres_correlations = correlations
        self.superres_reference = reference_image

        # Visualize reference image
        self.sr_reference_plot.clear()
        img_ref = pg.ImageItem(reference_image)
        self.sr_reference_plot.addItem(img_ref)

        # Visualize example correlation (detector pixel at +10, +10 from center)
        dy, dx = correlations.shape[2], correlations.shape[3]
        example_corr = correlations[:, :, dy//2 + 10, dx//2 + 10]
        self.sr_correlation_plot.clear()
        img_corr = pg.ImageItem(example_corr)
        self.sr_correlation_plot.addItem(img_corr)

        # Switch to Step 2 visualization
        self.sr_viz_stack.setCurrentIndex(2)

        # Enable Step 3
        self.sr_step3_btn.setEnabled(True)
        self.sr_step2_status.setText("✓ Cross-correlations computed")
        self.sr_step3_status.setText("Ready - click to compute shift maps")

        self.log_message(f"Super-res Step 2: Correlations computed, shape {correlations.shape}")

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Step 2 failed:\n{str(e)}")
        self.sr_step2_status.setText(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        self.sr_step2_btn.setEnabled(True)

def superres_step3_shifts(self):
    """Step 3: Compute shift maps"""
    if self.superres_correlations is None:
        return

    try:
        from ..processing.superres_processor import SuperResProcessor
        import numpy as np

        self.sr_step3_btn.setEnabled(False)
        self.sr_step3_status.setText("Computing shift maps...")
        QApplication.processEvents()

        processor = SuperResProcessor()

        # Compute shift maps
        xm_sub, ym_sub, im_sub = processor.find_shift_maps(
            self.superres_correlations, subpixel_refine=True
        )

        # Cache results
        self.superres_shift_maps = (xm_sub, ym_sub, im_sub)

        sy, sx = self.superres_data_cropped.shape[:2]

        # Center the shifts
        xm_centered = xm_sub - sy // 2
        ym_centered = ym_sub - sx // 2

        # === MAIN VISUALIZATION: QUIVER PLOT ===
        self.sr_quiver_plot.clear()

        # Subsample for clarity
        step = 4
        y_grid, x_grid = np.meshgrid(
            np.arange(0, xm_centered.shape[0], step),
            np.arange(0, xm_centered.shape[1], step),
            indexing='ij'
        )
        dy_sub = xm_centered[::step, ::step]
        dx_sub = ym_centered[::step, ::step]

        # Draw quiver plot using scatter + lines
        for i in range(dy_sub.shape[0]):
            for j in range(dy_sub.shape[1]):
                x0, y0 = x_grid[i, j], y_grid[i, j]
                dx, dy = dx_sub[i, j], dy_sub[i, j]
                # Scale arrows
                scale = 0.5
                arrow = pg.ArrowItem(angle=np.degrees(np.arctan2(dy, dx)),
                                    tipAngle=30, tailLen=scale*np.sqrt(dx**2+dy**2),
                                    pen='b', brush='b')
                arrow.setPos(x0, y0)
                self.sr_quiver_plot.addItem(arrow)

        # Inset visualizations
        self.sr_shifty_plot.clear()
        img_y = pg.ImageItem(xm_centered)
        self.sr_shifty_plot.addItem(img_y)

        self.sr_shiftx_plot.clear()
        img_x = pg.ImageItem(ym_centered)
        self.sr_shiftx_plot.addItem(img_x)

        self.sr_quality_plot.clear()
        img_q = pg.ImageItem(im_sub)
        self.sr_quality_plot.addItem(img_q)

        # Switch to Step 3 visualization
        self.sr_viz_stack.setCurrentIndex(3)

        # Display statistics
        stats_text = (f"Shift Y range: [{xm_centered.min():.1f}, {xm_centered.max():.1f}]\n"
                     f"Shift X range: [{ym_centered.min():.1f}, {ym_centered.max():.1f}]\n"
                     f"Quality: mean={im_sub.mean():.3f}, median={np.median(im_sub):.3f}\n"
                     f"Quality range: [{im_sub.min():.3f}, {im_sub.max():.3f}]")
        self.sr_step3_stats.setText(stats_text)

        # Enable Step 4
        self.sr_step4_btn.setEnabled(True)
        self.sr_step3_status.setText("✓ Shift maps computed")
        self.sr_step4_status.setText("Ready - click to reconstruct")

        self.log_message(f"Super-res Step 3: Shift maps computed")

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Step 3 failed:\n{str(e)}")
        self.sr_step3_status.setText(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        self.sr_step3_btn.setEnabled(True)

def superres_step4_reconstruct(self):
    """Step 4: Reconstruct super-resolution image"""
    if self.superres_shift_maps is None:
        return

    try:
        from ..processing.superres_processor import SuperResProcessor
        import numpy as np

        self.sr_step4_btn.setEnabled(False)
        self.sr_step4_status.setText("Reconstructing...")
        QApplication.processEvents()

        processor = SuperResProcessor()

        # Get parameters
        upscale = self.sr_upscale.value()
        det_radius = self.sr_det_radius.value()
        quality_thresh = self.sr_quality_thresh.value()

        xm_sub, ym_sub, im_sub = self.superres_shift_maps

        # Reconstruct
        superres, norm = processor.reconstruct_superres(
            self.superres_data_cropped,
            xm_sub, ym_sub, im_sub,
            fac=upscale,
            lims=det_radius,
            thresh=quality_thresh
        )

        # Standard BF for comparison
        w_y = self.superres_data_cropped.shape[2] // 2
        w_x = self.superres_data_cropped.shape[3] // 2
        std_bf = self.superres_data_cropped[:, :, w_y, w_x]

        # Cache results
        self.superres_results = {
            'superres_image': superres,
            'standard_bf': std_bf,
            'normalization': norm
        }

        # Visualize standard BF
        self.sr_std_bf_plot.clear()
        img_std = pg.ImageItem(std_bf)
        self.sr_std_bf_plot.addItem(img_std)
        self.sr_std_bf_plot.setTitle(f"Standard BF ({std_bf.shape[0]}×{std_bf.shape[1]})")

        # Visualize super-res BF
        self.sr_super_bf_plot.clear()
        img_super = pg.ImageItem(superres)
        self.sr_super_bf_plot.addItem(img_super)
        self.sr_super_bf_plot.setTitle(f"Super-Res BF ({superres.shape[0]}×{superres.shape[1]})")

        # Compute and display FFTs
        def compute_fft_log(image):
            fft = np.fft.fft2(image)
            fft_shifted = np.fft.fftshift(fft)
            fft_mag = np.abs(fft_shifted)
            return np.log(fft_mag + 1)

        std_fft = compute_fft_log(std_bf)
        self.sr_std_fft_plot.clear()
        img_std_fft = pg.ImageItem(std_fft)
        self.sr_std_fft_plot.addItem(img_std_fft)

        superres_fft = compute_fft_log(superres)
        self.sr_super_fft_plot.clear()
        img_super_fft = pg.ImageItem(superres_fft)
        self.sr_super_fft_plot.addItem(img_super_fft)

        # Switch to Step 4 visualization
        self.sr_viz_stack.setCurrentIndex(4)

        self.sr_step4_status.setText(f"✓ Reconstruction complete! ({upscale}× upscaling)")
        self.log_message(f"Super-res Step 4: Reconstruction complete, {superres.shape}")

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Step 4 failed:\n{str(e)}")
        self.sr_step4_status.setText(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        self.sr_step4_btn.setEnabled(True)
