import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import queue
import os
from fpdf import FPDF
from datetime import datetime
import io  # For in-memory image conversion

class PhotoDiffApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Difference Finder")
        self.root.geometry("1200x800")  # Fixed window size to fit three scaled panels

        # Variables to store images and states
        self.original_image = None
        self.comparison_image = None
        self.original_cv = None
        self.comparison_cv = None
        self.result_queue = queue.Queue()
        self.threshold = 30  # Fixed threshold, no longer adjustable via slider
        self.current_differences = []
        self.highlighted_result = None
        self.unchanged_comparison_image = None
        self.user_input = {}
        self.zoom_factors = {'original': 0.3, 'comparison': 0.3, 'highlighted': 0.3}  # Default zoom set to 0.3
        self.pan_starts = {'original': None, 'comparison': None, 'highlighted': None}  # Individual pan starts
        self.full_zoom_factors = {}  # Store zoom factors for full-image windows
        self.comparison_count = 0  # Track comparison count to debug alternating behavior

        # UI Setup
        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Top frame for buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Buttons and controls with modern styling
        upload_icon = tk.PhotoImage(file="upload_icon.png") if os.path.exists("upload_icon.png") else None  # Optional icon
        compare_icon = tk.PhotoImage(file="upload_icon.png") if os.path.exists("compare_icon.png") else None  # Optional icon (using upload_icon as placeholder, update if different)
        save_icon = tk.PhotoImage(file="save_icon.png") if os.path.exists("save_icon.png") else None  # Optional icon

        self.upload_original_btn = tk.Button(button_frame, text="Upload Original", command=self.upload_original, 
                                           bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat')
        self.upload_original_btn.pack(side=tk.LEFT, padx=5, pady=5)
        if upload_icon:
            self.upload_original_btn.config(image=upload_icon, compound=tk.LEFT)
            self.upload_original_btn.image = upload_icon

        self.upload_comparison_btn = tk.Button(button_frame, text="Upload Comparison", command=self.upload_comparison, 
                                             bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat')
        self.upload_comparison_btn.pack(side=tk.LEFT, padx=5, pady=5)
        if upload_icon:
            self.upload_comparison_btn.config(image=upload_icon, compound=tk.LEFT)
            self.upload_comparison_btn.image = upload_icon

        self.compare_btn = tk.Button(button_frame, text="Compare", command=self.compare_images_wrapper, 
                                   bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat')
        self.compare_btn.pack(side=tk.LEFT, padx=5, pady=5)
        if compare_icon:
            self.compare_btn.config(image=compare_icon, compound=tk.LEFT)
            self.compare_btn.config(text="")  # Remove text if using icon only
            self.compare_btn.image = compare_icon

        self.save_btn = tk.Button(button_frame, text="Save Result", command=self.save_result_as_pdf, 
                                bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat')
        self.save_btn.pack(side=tk.LEFT, padx=5, pady=5)
        if save_icon:
            self.save_btn.config(image=save_icon, compound=tk.LEFT)
            self.save_btn.image = save_icon

        self.export_btn = tk.Button(button_frame, text="Export Images", command=self.export_images, 
                                  bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat')
        self.export_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Removed the threshold_slider (sensitivity bar) from the UI

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, length=200, mode='determinate')
        self.progress.pack(side=tk.TOP, pady=5)

        # Frame for images (three panels side by side)
        image_frame = tk.Frame(main_frame)
        image_frame.pack(expand=True, fill=tk.BOTH)

        # Left frame for Original Image (dynamic scaled size)
        left_frame = tk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.original_label = tk.Label(left_frame, text="Original Image", font=('Helvetica', 12, 'bold'))
        self.original_label.pack(pady=5)
        self.original_canvas = tk.Canvas(left_frame, bg='white')
        self.original_canvas.pack(expand=True, fill=tk.BOTH)
        self.original_canvas.bind("<MouseWheel>", lambda e: self.zoom(e, 'original'))
        self.original_canvas.bind("<Button-1>", lambda e: self.start_pan(e, 'original'))
        self.original_canvas.bind("<B1-Motion>", lambda e: self.pan(e, 'original'))
        self.original_zoom_slider = tk.Scale(left_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL, label="Zoom", 
                                           command=lambda val: self.update_zoom('original', float(val)), 
                                           resolution=0.1, bg='#f0f0f0', troughcolor='#e0e0e0', 
                                           sliderlength=20, length=200)
        self.original_zoom_slider.set(0.3)  # Set default zoom to 0.3
        self.original_zoom_slider.pack(pady=5)
        self.original_view_btn = tk.Button(left_frame, text="View Full Image", command=lambda: self.open_full_image('original'), 
                                         bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat')
        self.original_view_btn.pack(pady=5)

        # Middle frame for Comparison Image (dynamic scaled size)
        middle_frame = tk.Frame(image_frame)
        middle_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.comparison_label = tk.Label(middle_frame, text="Comparison Image", font=('Helvetica', 12, 'bold'))
        self.comparison_label.pack(pady=5)
        self.comparison_canvas = tk.Canvas(middle_frame, bg='white')
        self.comparison_canvas.pack(expand=True, fill=tk.BOTH)
        self.comparison_canvas.bind("<MouseWheel>", lambda e: self.zoom(e, 'comparison'))
        self.comparison_canvas.bind("<Button-1>", lambda e: self.start_pan(e, 'comparison'))
        self.comparison_canvas.bind("<B1-Motion>", lambda e: self.pan(e, 'comparison'))
        self.comparison_zoom_slider = tk.Scale(middle_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL, label="Zoom", 
                                             command=lambda val: self.update_zoom('comparison', float(val)), 
                                             resolution=0.1, bg='#f0f0f0', troughcolor='#e0e0e0', 
                                             sliderlength=20, length=200)
        self.comparison_zoom_slider.set(0.3)  # Set default zoom to 0.3
        self.comparison_zoom_slider.pack(pady=5)
        self.comparison_view_btn = tk.Button(middle_frame, text="View Full Image", command=lambda: self.open_full_image('comparison'), 
                                           bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat')
        self.comparison_view_btn.pack(pady=5)

        # Right frame for Highlighted Changes (dynamic scaled size)
        right_frame = tk.Frame(image_frame)
        right_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.highlighted_label = tk.Label(right_frame, text="Highlighted Changes", font=('Helvetica', 12, 'bold'))
        self.highlighted_label.pack(pady=5)
        self.highlighted_canvas = tk.Canvas(right_frame, bg='white')
        self.highlighted_canvas.pack(expand=True, fill=tk.BOTH)
        self.highlighted_canvas.bind("<MouseWheel>", lambda e: self.zoom(e, 'highlighted'))
        self.highlighted_canvas.bind("<Button-1>", lambda e: self.start_pan(e, 'highlighted'))
        self.highlighted_canvas.bind("<B1-Motion>", lambda e: self.pan(e, 'highlighted'))
        self.highlighted_zoom_slider = tk.Scale(right_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL, label="Zoom", 
                                              command=lambda val: self.update_zoom('highlighted', float(val)), 
                                              resolution=0.1, bg='#f0f0f0', troughcolor='#e0e0e0', 
                                              sliderlength=20, length=200)
        self.highlighted_zoom_slider.set(0.3)  # Set default zoom to 0.3
        self.highlighted_zoom_slider.pack(pady=5)
        self.highlighted_view_btn = tk.Button(right_frame, text="View Full Image", command=lambda: self.open_full_image('highlighted'), 
                                            bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat')
        self.highlighted_view_btn.pack(pady=5)

        # Frame for differences list at the bottom
        diff_frame = tk.Frame(main_frame)
        diff_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.differences_listbox = tk.Listbox(diff_frame, height=10, width=80, bg='white', fg='#333333')
        self.differences_listbox.pack(expand=True, fill=tk.X)

        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.upload_original())
        self.root.bind('<Control-c>', lambda e: self.upload_comparison())
        self.root.bind('<Control-r>', lambda e: self.compare_images_wrapper())
        self.root.bind('<Control-s>', lambda e: self.save_result_as_pdf())

    def upload_original(self):
        # Prompt user to confirm clearing previous results
        if (self.original_image is not None or self.comparison_image is not None or 
            self.original_cv is not None or self.comparison_cv is not None or 
            self.current_differences or self.highlighted_result):
            if not messagebox.askyesno("Confirm Reset", "Uploading a new original image will clear all previous results. Continue?"):
                return  # Cancel the upload if user declines

        # Clear all previous results, including canvases, progress bar, and queue
        self.original_image = None
        self.comparison_image = None
        self.original_cv = None
        self.comparison_cv = None
        self.current_differences = []
        self.highlighted_result = None
        self.unchanged_comparison_image = None
        self.differences_listbox.delete(0, tk.END)
        self.original_canvas.delete("all")  # Clear the original canvas
        self.comparison_canvas.delete("all")  # Clear the comparison canvas explicitly
        self.highlighted_canvas.delete("all")  # Clear the highlighted canvas
        self.progress['value'] = 0  # Reset progress bar to 0
        self.result_queue = queue.Queue()  # Reset the result queue to clear any lingering data
        self.scale_and_display_images()  # Reset canvases to blank state with default zoom

        # Proceed with uploading the new original image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            try:
                img = Image.open(file_path).convert('RGB')
                if img.size[0] > 2000 or img.size[1] > 2000:
                    img.thumbnail((2000, 2000), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
                self.original_image = img
                self.original_cv = cv2.imread(file_path)  # Ensure OpenCV image is loaded immediately
                print(f"Uploaded original image: PIL={self.original_image is not None}, CV={self.original_cv is not None}")  # Debugging
                self.scale_and_display_images()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load original image: {e}")

    def upload_comparison(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Please upload an original image first!")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            try:
                img = Image.open(file_path).convert('RGB')
                if img.size[0] > 2000 or img.size[1] > 2000:
                    img.thumbnail((2000, 2000), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
                self.comparison_image = img
                self.comparison_cv = cv2.imread(file_path)  # Ensure OpenCV image is loaded immediately
                print(f"Uploaded comparison image: PIL={self.comparison_image is not None}, CV={self.comparison_cv is not None}")  # Debugging
                self.scale_and_display_images()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load comparison image: {e}")

    def scale_and_display_images(self):
        """Scale and display all three images to fit within the window without scrolling initially, but allow unlimited zooming."""
        # Get available space for images (subtract space for buttons, progress bar, and listbox)
        window_width = 1200 - 20  # Account for padding (10px on each side)
        window_height = 800 - 100  # Account for buttons (30px), progress bar (30px), and listbox (40px)

        # If both images are uploaded, scale them proportionally to fit side by side initially
        if self.original_image and self.comparison_image:
            orig_width, orig_height = self.original_image.size
            comp_width, comp_height = self.comparison_image.size

            # Calculate the maximum width for each panel (three panels side by side)
            max_panel_width = window_width // 3 - 10  # Subtract padding between panels (5px each side)
            max_panel_height = window_height - 60  # Subtract space for label (15px), zoom slider (30px), and view button (15px)

            # Find the scaling factor to fit both images within their respective panels while maintaining aspect ratio
            orig_scale = min(max_panel_width / orig_width, max_panel_height / orig_height, 1.0)
            comp_scale = min(max_panel_width / comp_width, max_panel_height / comp_height, 1.0)

            # Use the smaller scale to ensure all panels fit within the window initially, but apply default zoom of 0.3
            scale = min(orig_scale, comp_scale, 1.0) * 0.3  # Apply default zoom of 0.3

            # Resize images and update canvas sizes for initial display
            new_orig_width = int(orig_width * scale)
            new_orig_height = int(orig_height * scale)
            new_comp_width = int(comp_width * scale)
            new_comp_height = int(comp_height * scale)

            self.original_canvas.config(width=new_orig_width, height=new_orig_height)
            self.comparison_canvas.config(width=new_comp_width, height=new_comp_height)
            self.highlighted_canvas.config(width=new_comp_width, height=new_comp_height)  # Match comparison size for consistency

            # Display the scaled images, but keep highlighted canvas blank if no comparison has been performed
            self.display_image(self.original_image, self.original_canvas)
            self.display_image(self.comparison_image, self.comparison_canvas)
            if not self.highlighted_result:
                self.highlighted_canvas.delete("all")  # Keep highlighted canvas blank until comparison
            else:
                self.display_image(self.highlighted_result, self.highlighted_canvas)
        
        # If only one image is uploaded, scale it to fit its panel initially with default zoom of 0.3
        elif self.original_image:
            orig_width, orig_height = self.original_image.size
            max_panel_width = window_width // 3 - 10
            max_panel_height = window_height - 60
            scale = min(max_panel_width / orig_width, max_panel_height / orig_height, 1.0) * 0.3  # Apply default zoom of 0.3
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            self.original_canvas.config(width=new_width, height=new_height)
            self.display_image(self.original_image, self.original_canvas)
            self.comparison_canvas.delete("all")  # Clear comparison canvas explicitly
            self.highlighted_canvas.delete("all")  # Ensure highlighted canvas is blank
        
        elif self.comparison_image:
            comp_width, comp_height = self.comparison_image.size
            max_panel_width = window_width // 3 - 10
            max_panel_height = window_height - 60
            scale = min(max_panel_width / comp_width, max_panel_height / comp_height, 1.0) * 0.3  # Apply default zoom of 0.3
            new_width = int(comp_width * scale)
            new_height = int(comp_height * scale)
            self.comparison_canvas.config(width=new_width, height=new_height)
            self.display_image(self.comparison_image, self.comparison_canvas)
            self.original_canvas.delete("all")  # Clear original canvas explicitly
            self.highlighted_canvas.delete("all")  # Ensure highlighted canvas is blank

    def display_image(self, image, canvas):
        """Display the image, scaled based on current zoom, allowing zooming beyond canvas size with scrolling."""
        if image:
            # Get current canvas dimensions and zoom factor
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            canvas_type = self.get_canvas_type(canvas)
            zoom = self.zoom_factors[canvas_type]
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Calculate the scaled dimensions based on zoom, without limiting to canvas size
            new_width = int(img_width * zoom)
            new_height = int(img_height * zoom)
            
            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas.image = photo  # Keep reference
            canvas.config(scrollregion=canvas.bbox("all"))

    def get_canvas_type(self, canvas):
        """Determine the type of canvas (original, comparison, or highlighted)."""
        if canvas == self.original_canvas:
            return 'original'
        elif canvas == self.comparison_canvas:
            return 'comparison'
        elif canvas == self.highlighted_canvas:
            return 'highlighted'
        else:
            # For full-image canvases, we need to determine the type based on the image_type passed to open_full_image
            return self.get_image_type_from_window(canvas)  # Fallback or custom logic for full-image canvases

    def get_image_type_from_window(self, canvas):
        """Determine the image type based on the canvas in a full-image window."""
        for window in self.root.winfo_children():
            if isinstance(window, tk.Toplevel) and window.winfo_children():
                for widget in window.winfo_children():
                    if isinstance(widget, tk.Canvas) and widget == canvas:
                        # Check if the canvas has an image_type attribute (set in open_full_image)
                        return getattr(widget, 'image_type', None)
        return None  # Return None if the image type cannot be determined

    def align_images(self, original_cv, comparison_cv):
        original_gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
        comparison_gray = cv2.cvtColor(comparison_cv, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(original_gray, None)
        kp2, des2 = orb.detectAndCompute(comparison_gray, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        if len(matches) > 10:
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            aligned = cv2.warpPerspective(comparison_cv, H, (original_cv.shape[1], original_cv.shape[0]))
            return aligned
        return comparison_cv

    def detect_differences(self, original_cv, comparison_cv):
        differences = []
        height, width = original_cv.shape[:2]
        diff = cv2.absdiff(original_cv, comparison_cv)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Use fixed threshold of 30 (scaled to 0-255)
        scaled_threshold = int(self.threshold * 2.55)  # Convert 30 to 76.5, effectively 77 for OpenCV
        _, thresh = cv2.threshold(diff_gray, scaled_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Ignore small noise
                x, y, w, h = cv2.boundingRect(contour)
                diff_entry = (x, y, w, h, f"Difference at (x={x}, y={y}, width={w}, height={h})")
                differences.append(diff_entry)
        print(f"Detected differences with threshold={self.threshold} (scaled={scaled_threshold}), count={len(differences)}")  # Debugging
        return differences

    def compare_images(self, update_only=False):
        if self.original_cv is None or self.comparison_cv is None:
            messagebox.showerror("Error", "Please upload both images first!")
            self.result_queue.put(None)
            return

        try:
            # Debug: Check image shapes and comparison count
            print(f"Comparing images (count={self.comparison_count}) - Original CV shape: {self.original_cv.shape if self.original_cv is not None else 'None'}")
            print(f"Comparing images (count={self.comparison_count}) - Comparison CV shape: {self.comparison_cv.shape if self.comparison_cv is not None else 'None'}")

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Parallel alignment and difference detection
                future_aligned = executor.submit(self.align_images, self.original_cv, self.comparison_cv)
                comparison_aligned = future_aligned.result()

                if not update_only:
                    future_diff = executor.submit(self.detect_differences, self.original_cv, comparison_aligned)
                    differences = future_diff.result()
                else:
                    differences = self.detect_differences(self.original_cv, comparison_aligned)

            # Compute difference in color space with fixed threshold
            diff = cv2.absdiff(self.original_cv, comparison_aligned)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # Use fixed threshold of 30 (scaled to 0-255)
            scaled_threshold = int(self.threshold * 2.55)  # Convert 30 to 76.5, effectively 77 for OpenCV
            _, thresh = cv2.threshold(diff_gray, scaled_threshold, 255, cv2.THRESH_BINARY)

            # Create a colored highlight mask (red)
            highlight = np.zeros_like(comparison_aligned)
            highlight[:, :, 2] = thresh  # Red channel

            # Blend the highlight with the comparison image
            result = cv2.addWeighted(comparison_aligned, 0.7, highlight, 0.3, 0)

            # Convert to PIL Images for display and PDF
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(result_rgb)

            if not update_only:
                self.unchanged_comparison_image = Image.fromarray(cv2.cvtColor(self.comparison_cv, cv2.COLOR_BGR2RGB)).copy()
                # Store results and update UI immediately
                self.current_differences = differences
                self.highlighted_result = result_image
                diff_texts = [diff[4] for diff in differences]
                self.result_queue.put((result_image, diff_texts))
                # Force UI update to ensure progress bar and results are displayed
                self.root.update_idletasks()
                self.scale_and_display_images()
                self.differences_listbox.delete(0, tk.END)
                for diff in diff_texts:
                    self.differences_listbox.insert(tk.END, diff)
                messagebox.showinfo("Success", "Comparison completed. Use 'Save Result' to save as PDF.")
                self.comparison_count += 1  # Increment comparison count for debugging
            else:
                # Update only the highlighted canvas and differences list
                self.display_image(result_image, self.highlighted_canvas)
                self.differences_listbox.delete(0, tk.END)
                for diff in [d[4] for d in differences]:
                    self.differences_listbox.insert(tk.END, diff)

        except Exception as e:
            if not update_only:
                messagebox.showerror("Error", f"Comparison failed: {e}")
                print(f"Comparison error details (count={self.comparison_count}): {str(e)}")  # Debug: Log the error with count
                self.result_queue.put(None)

    def compare_images_wrapper(self):
        # Ensure both images are loaded before starting the comparison
        if self.original_image is None or self.comparison_image is None:
            messagebox.showerror("Error", "Please upload both original and comparison images first!")
            return

        # Debug: Check the current state of images and comparison count
        print(f"Before conversion (count={self.comparison_count}) - Original: PIL={self.original_image is not None}, CV={self.original_cv is not None}")
        print(f"Before conversion (count={self.comparison_count}) - Comparison: PIL={self.comparison_image is not None}, CV={self.comparison_cv is not None}")

        # Ensure OpenCV images are loaded immediately (should already be set from upload)
        if self.original_cv is None and self.original_image is not None:
            self.original_cv = self.pil_to_cv2(self.original_image)
            print(f"Converted Original CV (count={self.comparison_count}): {self.original_cv is not None}")
        if self.comparison_cv is None and self.comparison_image is not None:
            self.comparison_cv = self.pil_to_cv2(self.comparison_image)
            print(f"Converted Comparison CV (count={self.comparison_count}): {self.comparison_cv is not None}")

        # Debug: Verify state after conversion
        print(f"After conversion (count={self.comparison_count}) - Original: PIL={self.original_image is not None}, CV={self.original_cv is not None}")
        print(f"After conversion (count={self.comparison_count}) - Comparison: PIL={self.comparison_image is not None}, CV={self.comparison_cv is not None}")

        # Ensure OpenCV images exist and are valid before proceeding
        if self.original_cv is None or self.comparison_cv is None:
            messagebox.showerror("Error", "Failed to prepare images for comparison. Please try uploading again.")
            return

        # Force immediate processing by checking image validity
        try:
            if self.original_cv.shape is None or self.comparison_cv.shape is None:
                raise ValueError("Invalid image data detected")
        except AttributeError:
            messagebox.showerror("Error", "Invalid image data detected. Please try uploading again.")
            return

        # Reset progress bar and start comparison with immediate thread execution
        self.progress['value'] = 0
        compare_thread = Thread(target=self.compare_images_with_progress, daemon=True)
        compare_thread.start()
        self.root.after_idle(self.check_result)  # Use after_idle for immediate processing

        # Debug: Ensure thread starts immediately
        print(f"Thread started for comparison (count={self.comparison_count}), is_alive: {compare_thread.is_alive()}")

    def compare_images_with_progress(self):
        try:
            for i in range(100):
                import time
                time.sleep(0.01)  # Maintain short sleep for smooth progress
                self.result_queue.put(i)
            self.compare_images()
            self.result_queue.put(None)  # Signal completion
        except Exception as e:
            print(f"Error in compare_images_with_progress (count={self.comparison_count}): {str(e)}")
            self.result_queue.put(None)

    def check_result(self):
        try:
            value = self.result_queue.get(timeout=0.05)  # Reduced timeout for faster queue checks
            if value is not None:
                if isinstance(value, int):
                    self.progress['value'] = value
                    self.root.after(5, self.check_result)  # Faster update for smoother progress
                else:
                    result_image, differences = value
                    if result_image is not None:
                        self.scale_and_display_images()  # Ensure all images are scaled and displayed, including highlighted
                        self.differences_listbox.delete(0, tk.END)
                        for diff in differences:
                            self.differences_listbox.insert(tk.END, diff)
                        messagebox.showinfo("Success", "Comparison completed. Use 'Save Result' to save as PDF.")
                    else:
                        messagebox.showerror("Error", "Comparison failed.")
            else:
                self.progress['value'] = 100
        except queue.Empty:
            self.root.after(5, self.check_result)  # Faster update for smoother progress

    def save_result_as_pdf(self):
        if self.original_image is None or self.highlighted_result is None or self.unchanged_comparison_image is None:
            messagebox.showerror("Error", "Please upload, compare, and highlight images first!")
            return

        # Prompt for user input
        self.user_input = {}
        input_dialog = tk.Toplevel(self.root)
        input_dialog.title("Enter PDF Information")
        input_dialog.geometry("400x400")
        input_dialog.transient(self.root)
        input_dialog.grab_set()

        tk.Label(input_dialog, text="Enter your information for the PDF:").pack(pady=5)

        tk.Label(input_dialog, text="Your Name:").pack()
        name_entry = tk.Entry(input_dialog)
        name_entry.pack(pady=5)

        tk.Label(input_dialog, text="Image Name:").pack()
        image_name_entry = tk.Entry(input_dialog)
        image_name_entry.pack(pady=5)

        tk.Label(input_dialog, text="Image Summary:").pack()
        image_summary_entry = tk.Text(input_dialog, height=3, width=30)
        image_summary_entry.pack(pady=5)

        tk.Label(input_dialog, text="Image Location:").pack()
        image_location_entry = tk.Entry(input_dialog)
        image_location_entry.pack(pady=5)

        def save_input():
            self.user_input = {
                "name": name_entry.get().strip() or "Not provided",
                "image_name": image_name_entry.get().strip() or "Not provided",
                "image_summary": image_summary_entry.get("1.0", tk.END).strip() or "Not provided",
                "image_location": image_location_entry.get().strip() or "Not provided"
            }
            input_dialog.destroy()
            self._save_pdf_after_input()

        tk.Button(input_dialog, text="Save", command=save_input, bg='#2ecc71', fg='white', font=('Helvetica', 10, 'bold'), relief='flat').pack(pady=10)
        tk.Button(input_dialog, text="Cancel", command=input_dialog.destroy, bg='#e74c3c', fg='white', font=('Helvetica', 10, 'bold'), relief='flat').pack(pady=5)

        input_dialog.wait_window()

    def _save_pdf_after_input(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if not file_path:
            return

        try:
            if not file_path.endswith('.pdf'):
                file_path += '.pdf'
            print(f"Saving PDF to: {file_path}")

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Page 1: Title Page with User Input and Date/Time
            pdf.add_page()
            pdf.set_font("Arial", size=24, style='B')
            pdf.cell(0, 10, "Photo Comparison Results", ln=True, align="C")
            pdf.set_font("Arial", size=12)
            comparison_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf.cell(0, 10, f"Comparison Date and Time: {comparison_time}", ln=True, align="C")
            pdf.cell(0, 10, f"User Name: {self.user_input['name']}", ln=True, align="C")
            pdf.cell(0, 10, f"Image Name: {self.user_input['image_name']}", ln=True, align="C")
            pdf.multi_cell(0, 10, f"Image Summary: {self.user_input['image_summary']}", align="C")
            pdf.cell(0, 10, f"Image Location: {self.user_input['image_location']}", ln=True, align="C")

            # Page 2: Original Image with title
            original_pil = self.original_image.copy()
            original_pil.thumbnail((500, 700), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            original_path = "temp_original.png"
            original_pil.save(original_path, "PNG")
            pdf.add_page()
            pdf.set_font("Arial", size=16, style='B')
            pdf.cell(0, 10, "Original Image", ln=True, align="C")
            pdf.image(original_path, x=10, y=20, w=190)

            # Page 3: Comparison Image (before highlights) with title
            unchanged_comparison_pil = self.unchanged_comparison_image.copy()
            unchanged_comparison_pil.thumbnail((500, 700), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            unchanged_path = "temp_comparison.png"
            unchanged_comparison_pil.save(unchanged_path, "PNG")
            pdf.add_page()
            pdf.set_font("Arial", size=16, style='B')
            pdf.cell(0, 10, "Comparison Image", ln=True, align="C")
            pdf.image(unchanged_path, x=10, y=20, w=190)

            # Page 4: Comparison Image with Highlights and title
            highlighted_pil = self.highlighted_result.copy()
            highlighted_pil.thumbnail((500, 700), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            highlighted_path = "temp_highlighted.png"
            highlighted_pil.save(highlighted_path, "PNG")
            pdf.add_page()
            pdf.set_font("Arial", size=16, style='B')
            pdf.cell(0, 10, "Comparison Image with Highlights", ln=True, align="C")
            pdf.image(highlighted_path, x=10, y=20, w=190)

            # Page 5: List of Changes with title
            pdf.add_page()
            pdf.set_font("Arial", size=16, style='B')
            pdf.cell(0, 10, "List of Changes", ln=True, align="C")
            text = "\n".join([diff[4] for diff in self.current_differences])
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, text)

            pdf.output(file_path)
            messagebox.showinfo("Success", f"Result saved as {file_path}")

            if os.path.exists(original_path): os.remove(original_path)
            if os.path.exists(unchanged_path): os.remove(unchanged_path)
            if os.path.exists(highlighted_path): os.remove(highlighted_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save PDF: {e}\nDetails: {str(e)}")

    def export_images(self):
        if self.original_image and self.comparison_image and self.highlighted_result:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if output_dir:
                try:
                    self.original_image.save(os.path.join(output_dir, "original.png"), "PNG")
                    self.comparison_image.save(os.path.join(output_dir, "comparison.png"), "PNG")
                    self.highlighted_result.save(os.path.join(output_dir, "highlighted.png"), "PNG")
                    messagebox.showinfo("Success", "Images exported successfully.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export images: {e}")
        else:
            messagebox.showerror("Error", "Please upload and compare images first!")

    def update_zoom(self, canvas_type, value):
        """Update the zoom factor and redraw the image for the specified canvas, allowing unlimited zooming with scrolling."""
        self.zoom_factors[canvas_type] = value
        self.zoom_factors[canvas_type] = max(0.1, min(10.0, self.zoom_factors[canvas_type]))  # Clamp between 0.1 and 10.0
        canvas = {
            'original': self.original_canvas,
            'comparison': self.comparison_canvas,
            'highlighted': self.highlighted_canvas
        }[canvas_type]
        self.display_image({
            'original': self.original_image,
            'comparison': self.comparison_image,
            'highlighted': self.unchanged_comparison_image if canvas_type == 'highlighted' and not self.highlighted_result else self.highlighted_result
        }[canvas_type], canvas)
        canvas.config(scrollregion=canvas.bbox("all"))
        # Update the corresponding zoom slider
        self.update_zoom_slider(canvas_type)

    def update_zoom_slider(self, canvas_type):
        """Update the zoom slider value to match the current zoom factor."""
        slider = {
            'original': self.original_zoom_slider,
            'comparison': self.comparison_zoom_slider,
            'highlighted': self.highlighted_zoom_slider
        }[canvas_type]
        slider.set(self.zoom_factors[canvas_type])

    def zoom(self, event, canvas_type):
        canvas = {
            'original': self.original_canvas,
            'comparison': self.comparison_canvas,
            'highlighted': self.highlighted_canvas
        }[canvas_type]
        scale = 1.1 if event.delta > 0 else 1/1.1
        self.zoom_factors[canvas_type] *= scale
        self.zoom_factors[canvas_type] = max(0.1, min(10.0, self.zoom_factors[canvas_type]))  # Clamp between 0.1 and 10.0
        self.update_zoom(canvas_type, self.zoom_factors[canvas_type])
        # Update the corresponding zoom slider
        self.update_zoom_slider(canvas_type)

    def start_pan(self, event, canvas_type):
        self.pan_starts[canvas_type] = (event.x, event.y)
        canvas = {
            'original': self.original_canvas,
            'comparison': self.comparison_canvas,
            'highlighted': self.highlighted_canvas
        }[canvas_type]
        canvas.scan_mark(event.x, event.y)

    def pan(self, event, canvas_type):
        if self.pan_starts[canvas_type]:
            canvas = {
                'original': self.original_canvas,
                'comparison': self.comparison_canvas,
                'highlighted': self.highlighted_canvas
            }[canvas_type]
            canvas.scan_dragto(event.x, event.y, gain=1)
            self.pan_starts[canvas_type] = (event.x, event.y)

    def open_full_image(self, image_type):
        """Open a new window to display the full image with scrolling, zooming, panning, and a zoom slider, allowing unlimited zooming."""
        image = {
            'original': self.original_image,
            'comparison': self.comparison_image,
            'highlighted': self.highlighted_result
        }[image_type]
        if not image:
            messagebox.showerror("Error", f"No {image_type} image available to view!")
            return

        # Create a new window
        full_window = tk.Toplevel(self.root)
        full_window.title(f"Full {image_type.capitalize()} Image")
        full_window.geometry("800x600")  # Default size for the new window

        # Main frame for the full window
        full_frame = tk.Frame(full_window)
        full_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Canvas for the full image
        full_canvas = tk.Canvas(full_frame, bg='white')
        full_canvas.pack(expand=True, fill=tk.BOTH)

        # Store the image type in the canvas for later reference
        full_canvas.image_type = image_type  # Add image_type to the canvas object

        # Image dimensions and initial zoom factor
        img_width, img_height = image.size
        full_zoom_factor = 1.0  # Use a local variable for this window's zoom state
        full_pan_start = None  # Initialize pan start variable locally

        def display_full_image():
            # Calculate scaled dimensions based on zoom, without limiting to canvas size
            new_width = int(img_width * full_zoom_factor)
            new_height = int(img_height * full_zoom_factor)
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            full_canvas.delete("all")
            full_canvas.create_image(0, 0, anchor="nw", image=photo)
            full_canvas.image = photo  # Keep reference
            full_canvas.config(scrollregion=full_canvas.bbox("all"))

        display_full_image()

        # Add scrollbars
        h_scroll = tk.Scrollbar(full_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        h_scroll.config(command=full_canvas.xview)
        full_canvas.config(xscrollcommand=h_scroll.set)

        v_scroll = tk.Scrollbar(full_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        v_scroll.config(command=full_canvas.yview)
        full_canvas.config(yscrollcommand=v_scroll.set)

        # Zoom slider
        full_zoom_slider = tk.Scale(full_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL, label="Zoom", 
                                  command=lambda val: self.update_full_zoom(full_canvas, float(val), img_width, img_height, full_zoom_factor), 
                                  resolution=0.1, bg='#f0f0f0', troughcolor='#e0e0e0', 
                                  sliderlength=20, length=200)
        full_zoom_slider.set(1.0)
        full_zoom_slider.pack(side=tk.BOTTOM, pady=5)

        # Zoom and pan bindings
        def full_zoom(e):
            global full_zoom_factor  # Use global to modify the local variable within this scope
            scale = 1.1 if e.delta > 0 else 1/1.1
            full_zoom_factor *= scale
            full_zoom_factor = max(0.1, min(10.0, full_zoom_factor))  # Clamp between 0.1 and 10.0
            self.update_full_zoom(full_canvas, full_zoom_factor, img_width, img_height, full_zoom_factor)
            full_zoom_slider.set(full_zoom_factor)

        def full_start_pan(e):
            nonlocal full_pan_start  # Use nonlocal to modify the local variable in the enclosing scope
            full_pan_start = (e.x, e.y)
            full_canvas.scan_mark(e.x, e.y)

        def full_pan(e):
            nonlocal full_pan_start  # Use nonlocal to access and modify the local variable
            if full_pan_start:
                full_canvas.scan_dragto(e.x, e.y, gain=1)
                full_pan_start = (e.x, e.y)

        full_canvas.bind("<MouseWheel>", full_zoom)
        full_canvas.bind("<Button-1>", full_start_pan)
        full_canvas.bind("<B1-Motion>", full_pan)

    def update_full_zoom(self, canvas, value, img_width, img_height, full_zoom_factor):
        """Update the zoom factor and redraw the image in the full image window, allowing unlimited zooming."""
        full_zoom_factor = value  # Update the local variable
        full_zoom_factor = max(0.1, min(10.0, full_zoom_factor))  # Clamp between 0.1 and 10.0
        # Calculate scaled dimensions based on zoom, without limiting to canvas size
        new_width = int(img_width * full_zoom_factor)
        new_height = int(img_height * full_zoom_factor)
        # Get the image type directly from the canvas
        image_type = getattr(canvas, 'image_type', None)
        if image_type:
            image = self.get_image_for_type(image_type)
            if image:  # Ensure the image exists before resizing
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
                photo = ImageTk.PhotoImage(resized_image)
                canvas.delete("all")
                canvas.create_image(0, 0, anchor="nw", image=photo)
                canvas.image = photo  # Keep reference
                canvas.config(scrollregion=canvas.bbox("all"))
            else:
                messagebox.showerror("Error", f"No {image_type} image available for zooming.")
        else:
            messagebox.showerror("Error", "Image type not available for zooming. Please ensure an image is loaded.")

    def get_image_for_type(self, image_type):
        """Get the appropriate image based on the type."""
        return {
            'original': self.original_image,
            'comparison': self.comparison_image,
            'highlighted': self.highlighted_result
        }.get(image_type, None)  # Return None if image_type is not found

    def pil_to_cv2(self, pil_image):
        """Convert PIL image to OpenCV format in-memory."""
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        return cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)

if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoDiffApp(root)
    root.mainloop()
