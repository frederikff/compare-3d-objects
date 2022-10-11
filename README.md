# compare-3d-objects

Requirements: \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; trimesh~=3.12.9 \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; numpy~=1.23.1 \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; pandas~=1.4.3 \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; pyvista~=0.36.1 \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; matplotlib~=3.5.2 \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; plotly~=5.9.0 \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; numpy-stl~=2.17.1 \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; pyntcloud~=0.3.1

<br>

Execution:
- Running "compare_models_final.py" works without settings-adjustment (the results will be printed in the terminal)
- Command-line execution allows adjustment of settings: 
  python compare_models_final.py input_1.stl(or .csv) input2.stl(or .csv) --options
- Options: \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --scale x(integer between 100 and 1.000.000, default 100.000) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -> target-scaling-volume \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --density x(integer between 1 and 5 for x and 1 and 3 for y, default 3) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -> illustration-variable: every xth point is displayed \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --transparency x.x(foating point number between 0.0 and xy, default 0.0 is automatic-mode) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -> variable of the illustration-appearance \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --show x(integer between 0 and 3, default 1) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -> select the illustration-method: 1=pyvista-plotter 2=matplotlib 3=plotly 0=none \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --export x(integer between 0 and 1, default 0) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -> yes=1 no=0 for exporting the object matrix, the similarity results and the illustration \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --boundingbox (integer between 0 and 1, default 1) \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -> yes=1 no=0 for using the bounding box in the normalisation process
- Running compare_models_interface_final.py provides user-interface
- [donwload .exe application of the user-interface script form google drive](https://drive.google.com/file/d/14y_kFhS_WtN7LKfyKwF03NxBgSiKPPL6/view?usp=sharing)

<br>

Information:
- Scaling-target-volume: reference-volume, that both objects are scaled to before comparison 
- Bounding box: additional normalisation-method to extend the inertia-method
- Pyvista-plotter additionally provides visualisation of the volume-similarity
- Transparency=0.0 starts the automatic-mode (optimized for scaling target-volume of 100.000 and resolution of 1920x1080)
- Recommended setting for matplotlib: Density 3, Transparency 0.05-0.2
						  Density 2, Transparency 0.04-0.1
						  Density 1, Transparency 0.01-0.02
- Recommended settings for plotly: Density 1-3
					     Transparency 0.06-0.2
- Export-matrix: 3D numpy arrays of 0s(no object), 1s(only first object), 2s(only second object) and 3s(both objects) as .csv files
- Export results: calculated similarity values as .txt file


Restrictions:
- The accuracy increases with the target volume
- The accuracy decreases for .csv objects because of the convex hull that is needed for conversion
- The computation-time increases with the target-volume and with usage of the bounding-box normalisation
- The pose-normalisation is not always optimal, which has a high impact on the results
