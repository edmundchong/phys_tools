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
    
    
2. __Unit.__<br> 
    This contains methods that interact with individual units. Again, there are several subclasses
    
    1. session - the Session object from which the unit is recorded (including all stimulus and related 
    metadata.)
    1. get_rasters_samples()
    1. plot_rasters()
    1. plot_psth_time()
    1. _...other stimulus specific plotters and getters_