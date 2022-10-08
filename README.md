# compare-3d-objects

Information:
- Volume-similarity = volume of the smaller object
				  / sum of both volumes
- Surface-area-similarity = surface-area of the smaller object
					   / sum of both surface-areas
- Shape-similarity = overlapping volume(after the objects where normalised and after the 
								smaller object was scaled to the same size of the
								larger object) 
				 / sum of both volumes


Execution:
- Command-line: python compare_models_7.py input_1.stl(or csv) input2.stl(or csv) --options

- Run: only possible without options


Options:
--density: int between1 and 5 (every Xth point is displayed)
--transparency: float between 0.02 and 0.5 (see-through characteristics)
--show: int between 0 and 3 (0=no illustration, 
					    1=illustration and png-export with matplotlib,
					    2=illustration and html-export with plotly, 
					    3=illustration with pyvista plotter)
--export_matrix: int between 0 and 1 (0=do not export comparison matrix
							   1=export comparison matrix)
--export_results: int between 0 and 1 (0=do not export results
							    1=export results as txt)

Options-information:
- Recommended setting for matplotlib: density 1-4, transparency 0.02-0.1
- Recommended settings for plotly: density 1-3, Transparency 0.06-0.2
- Matplotlib-illustration also changes with similarity
- Export-matrices: 3D numpy arrays of 0s(no object),
							   1s(only first object),
							   2s(only second object),
							   3s(both objects) as .csv files
- Export results: calculated similarity values as .txt file'
