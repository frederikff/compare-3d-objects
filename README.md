# compare-3d-objects

Requirements:
	trimesh~=3.12.9
	numpy~=1.23.1
	pandas~=1.4.3
	pyvista~=0.36.1
	matplotlib~=3.5.2
	plotly~=5.9.0
	numpy-stl~=2.17.1
	pyntcloud~=0.3.1


Execution:
- Running "compare_models_final.py" works without settings-adjustment
- The results will be printed in the terminal
- Command-line execution allows adjustment of settings: 
  python compare_models_final.py input_1.stl(or .csv) input2.stl(or .csv) --options
- Options: --scale x(integer between 100 and 1.000.000, default 100.000) 
		 -> target-scaling-volume
	     --density x(integer between 1 and 5 for x and 1 and 3 for y, default 3) 
		 -> illustration-variable: every xth point is displayed
	     --transparency x.x(foating point number between 0.0 and xy, default 0.0 is automatic-mode) 
		 -> variable of the illustration-appearance
	     --show x(integer between 0 and 3, default 1) 
		 -> select the illustration-method: 1=pyvista-plotter 2=matplotlib 3=plotly 0=none
	     --export x(integer between 0 and 1, default 0) 
		 -> yes=1 no=0 for exporting the object matrix, the similarity results and the illustration
	     --boundingbox (integer between 0 and 1, default 1) 
		 -> yes=1 no=0 for using the bounding box in the normalisation process
- Running compare_models_interface_final.py provides user-interface


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
