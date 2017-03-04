Models
===

This package contains the data models for electrophysiology data. These models are intended to be used either
in interactive sessions or to be used as data models the attached gui package ("views").

Each module contains (at least) two different models:
1. __Session.__<br>
    This contains methods that load store and retrieve data related to an entire 
     electrophysiology recording session. There are several subclasses for different types of recording sessions
     (i.e. OdorSession, PatternSession). Session subclasses are associated with specific unit types that
     are designed to interact with specific stimuli structures.
 
   Each subclass of Session contains the following attributes and methods:
    1. stimuli - structure containing the stimuli presented during the recording session.
    1. units() - returns list of unit objects contained in the session
    1. millis_to_samples() - converts between timebases
    1. samples_to_millis()
    1. fs - sampling frequency of the recording (in Hz)
    
    
    
2. __Unit.__<br> 
    This contains methods that interact with individual units. Again, there are several subclasses
    
    1. session - the Session object from which the unit is recorded (including all stimulus and related 
    metadata.)
    1. get_rasters_samples()
    1. plot_rasters()
    1. plot_psth_time()
    1. _...other stimulus specific plotters and getters_
    
##Basic use
To load a session and get a list of units:
```python
import phys_tools as pt
import matplotlib.pyplot as plt
s = pt.models.OdorSession("your_path/your_filename.dat")  # this is specific for odor units.
all_units = s.units()  # type: list
units_above_rating_3 = s.units_gte(3)
```
To retrieve spikes within a specfic epoch:
```python
unit = s.units(1)  # type: OdorUnit

unit.get_epoch_ms(500, 1000)  # returns an array of all spikes between 500 and 1000 ms. 
# Times are relative to start. Ie a spike at 501 ms of the recording is represented in the array as 1.
```

To retrieve spikes (or "rasters") related to a specific stimulus:
```python
odor = 'pinene'
conc = .001
pretime = 100  # number of ms prior to stimulus
posttime = 500 # nubmer of ms after inhalation to return
rasters = unit.get_odor_rasters(odor, conc, pretime, posttime)
trials, times, shape = rasters
# you can plot like this:

plt.scatter(times, trials); plt.show()
```
The trials and times arrays are each of length equal to the total number of spikes across all trials.
So for the ith spike, the trial number is stored as `trials[i]`, and its time is stored as `times[i]`.

You can also just call functions to plot this using functions built into the classes:
```python
unit.plot_odor_rasters(odor, conc, pretime, posttime); plt.show()
unit.plot_odor_psth(odor, conc, pretime, posttime); plt.show()
```

To plot unit information that is not stimulus related:
```python
unit.plot_autocorrelation()
unit.plot_template()
```
