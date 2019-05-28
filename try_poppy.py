
# coding: utf-8

# In[1]:


import poppy


# In[8]:


osys = poppy.OpticalSystem()
osys.add_pupil( poppy.CircularAperture(radius=3))    # pupil radius in meters
osys.add_detector(pixelscale=0.010, fov_arcsec=5.0)  # image plane coordinates in arcseconds

psf = osys.calc_psf(2e-6)                            # wavelength in microns
# poppy.display_PSF(psf, title='The Airy Function') 


# In[6]:


values

