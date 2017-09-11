# Massively parallel multi-electrode recordings of macaque motor cortex during an instructed delayed reach-to-grasp task

## Summary
We provide two electrophysiological datasets recorded via a 10-by-10 multi-electrode array chronically implanted in the motor cortex of two macaque monkeys during an instructed delayed reach-to-grasp task. The datasets contain the continuous measure of extracellular potentials at each electrode sampled at 30 kHz, the local field potentials sampled at 1 kHz and the timing of the online and offline extracted spike times. It also includes the timing of several task-related and behavioral events recorded along with the electrophysiological data. Finally, the datasets provide a complete set of metadata structured in a standardized format. These metadata allow easy access to detailed information about the datasets such as the settings of the recording hardware, the array specifications, the location of the implant in the motor cortex, information about the monkeys, or the offline spike sorting.
The two datasets can be exploited to address crucial issues in neurophysiology such as: What are the principles of neural interactions in a local cortical network and how are these interactions modulated during a well-described behavioral task?  How different neuronal signals such as single-unit activity, multi-unit activity or LFPs relate to each other? Which spike sorting methods provide the best estimate of single unit activity?  

## Repository structure

### Directory datasets
Contains the two data sets `i140703-001` and `l101210-001`. Original data files are provided in the Blackrock file format (.nev, .ns2, .ns5, .ns6, .ccf), e.g., `i140703-001.nev`, `i140703-001.ns6`,.... The files `i140703-001-03.nev` and `l101210-001-02.nev` contain offline spike sorted data for both datasets as opposed to the original recordings `i140703-001.nev` and `l101210-001.nev` which contain the same spikes, but unreliable sorting that should not be used. The files `i140703-001.odml` and `l101210-001.odml` contain extensive metadata describing the datasets in the odML format. The Excel files `i140703-001.xls` and `l101210-001.xls` contain the same information as in the odML for easy reading and browsing, however, they are not used by the loading routines. The odml.xsl is an XML schema that is required for viewing the odML files with a web browser.

### Directory datasets_matlab
Contains the data and metadata output of the Python loading routines in the MATLAB .mat file format. These files are provided for convenience for MATLAB users, however, note that these files are not the original data files and contain a condensed, interpreted subset of the original data. Due to size restrictions of the MATLAB file format, the files `i140703-001_lfp-spikes.mat` and `l101210-001 _lfp-spikes.mat` contain only spikes and LFP data (for monkey N), while raw data is saved separately for each channel in correspondingly named files.

### Directory code
Contains example code to help in loading and analyzing the data. The file `examply.py` is a Python script that acts as a tutorial for loading and plotting data. The scripts `data_overview_1.py` and `data_overview_2.py` reproduce the plots of the data found  in the publication. The files `neo_utils.py` and `odml_utils.py` contain useful utility routines to work with data and metadata. Finally, the file `example.m` contains a rudimentary MATLAB script demonstrating how to use the data provided in the .mat files.

To run the Python example code, download the release of this repository, and install the requirements in `code/requirements.txt`. Then, run the example via
```
   cd code
   python example.py
```
The script produces a figure saved in three different graphics file formats.

### Directory code/reachgraspio
Contains the file `reachgraspio.py`, which contains the loading routine specific to the Reach-to-Grasp experiments in this repository. This loading routine merges the recorded data with metadata information from the odML files into a common Neo object. It is recommended that this loading routine is used in combination with the odML and Neo libraries (see below) to work on the data.

### Further subdirectories of code
The subdirectories `python-neo`, `python-odml`, and `elephant` contain snapshots of the Neo[1], odML[2], and Elephant[3] libraries, respectively, that are required by the example scripts and the reachgraspio loading routine. In short, Neo provides the data model, generic Blackrock loading routines, and APIs used to load the data; odML provides an API to handle the metadata files; and Elephant is a library for the analysis of neuronal data based on the Neo data model that is used by the example script for filtering raw signals to obtain offline filtered LFPs. By modifying the file `load_local_neo_odml_elephant.py` in the code directory it is possible to instruct the example scripts to use system-wide installed versions of these libraries instead of the static snapshots. Note however, that future versions of these libraries may requires adapted versions of the `reachgraspio.py` loading routine (see Updates below).
* [1] https://github.com/NeuralEnsemble/python-neo
* [2] https://github.com/G-Node/python-odml
* [3] https://github.com/NeuralEnsemble/elephant

## Updates
Updated versions of the codes will be provided at:
https://web.gin.g-node.org/INT/multielectrode_grasp
This includes, in particular, the loading routine reachgraspio.py, which may need to be adapted as new versions of the Neo and odML libraries become available.

## Related Publications
* Riehle, A., Wirtssohn, S., Grün, S., & Brochier, T. (2013). Mapping the spatio-temporal structure of motor cortical LFP and spiking activities during reach-to-grasp movements. Frontiers in Neural Circuits, 7, 48. https://doi.org/10.3389/fncir.2013.00048
* Milekovic, T., Truccolo, W., Grün, S., Riehle, A., & Brochier, T. (2015). Local field potentials in primate motor cortex encode grasp kinetic parameters. NeuroImage, 114, 338–355. https://doi.org/10.1016/j.neuroimage.2015.04.008
* Torre, E., Quaglio, P., Denker, M., Brochier, T., Riehle, A., & Grun, S. (2016). Synchronous spike patterns in macaque motor cortex during an instructed-delay reach-to-grasp task. Journal of Neuroscience, 36(32), 8329–8340. https://doi.org/10.1523/JNEUROSCI.4375-15.2016
* Zehl, L., Jaillet, F., Stoewer, A., Grewe, J., Sobolev, A., Wachtler, T., Brochier, T., Riehle, A., Denker, M., & Grün, S. (2016). Handling Metadata in a Neurophysiology Laboratory. Frontiers in Neuroinformatics, 10, 26. https://doi.org/10.3389/fninf.2016.00026
* Denker, M., Zehl, L., Kilavik, B. E., Diesmann, M., Brochier, T., Riehle, A., & Grün, S. (2017). LFP beta amplitude is predictive of mesoscopic spatio-temporal phase patterns, 1703.09488 [q-NC]. https://arxiv.org/abs/1703.09488

## Licensing
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Massively parallel multi-electrode recordings of macaque motor cortex during an instructed delayed reach-to-grasp task</span> in the directories `datasets` and `datasets_matlab` by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Institut de Neurosciences de la Timone (INT), UMR 7289, CNRS – Aix Marseille Université, Marseille, France and Institute of Neuroscience and Medicine (INM-6), Forschungszentrum Jülich, Jülich, Germany</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

All code in the directories `code`, `code/python-odml`, `code/python-neo`, `code/elephant`, and `code/reachgraspio` are each published under the BSD 3 clause licenses. See the `LICENSE.txt` or `LICENSE` files in the corresponding directories for the full license.

