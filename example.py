from raysttok import RaySTTOK


####################################################################################################################
# Ray-tracing simulation of lamp, synthethic plasma profile or single emissive pixel

raysttok = RaySTTOK(reflections=True, pixel_samples=100)

# raysttok.place_lamp(radius=0.05, angle=95)
raysttok.place_plasma(shape='bessel', emissiv_gain=1e3, mean=[0.05, -0.05], cov=[[5e-4, 0], [0, 5e-4]])
# raysttok.place_single_pixel(pixel_side=60, pixel_id=2145)

raysttok.simulate_rays()
raysttok.plot_detectors()
raysttok.show_plots()


####################################################################################################################
# Computation of projection matrix through tay-tracing simulation of individual pixels 

raysttok = RaySTTOK(reflections=True, pixel_samples=10)
raysttok.get_proj_matrix(pixel_side=15, out_file="proj_matrix1")
raysttok.plot_line_matrix(line=0, mat_file="proj_matrix1.npy")
raysttok.show_plots()


