# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:25:45 2023

@author: dl923 / leadbot
"""

#%% Imports
import os
import argparse
from Bio.PDB import PDBParser
import mdtraj as md
import time
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import trange
from keras.models import load_model
import joblib
import traceback

try:
    pd.set_option("future.no_silent_downcasting", True)
except:
    pass

#%% Efficient script with metadata harvest
def calculate_atom_distance(residue1, residue2, atom1, atom2):
    """
    Calculate linear angstrom distance between select atom in pair of residues (X, Y)

    Parameters
    ----------
    residue1 : residue X
    residue2 : residue Y
    atom1: atom target for residue1
    atom2: atom target for residue2

    Returns
    -------
    int: angstrom distance between atom1 and atom2

    """
    A1 = residue1[atom1]
    A2 = residue2[atom2]
    distance=A1-A2
    return distance

def count_beta_strands(dssp_structure, minimum_threshold):
    """
    Parameters
    ----------
    dssp_structure : dssp structure calculated with mdtraj
    minimum_threshold : minimum length for beta strand to be considered

    Returns
    -------
    number_strands : number of beta strands
    mean_len_strands : mean length of beta strands

    """
    i=0
    blocks=[]
    block_len=[]
    current_strand=[]
    for ss in dssp_structure:
        if ss=='E': # extended beta strand in DSSP classification see https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html
            i+=1
            current_strand.append(ss)
        #if next ss is not a beta strand, break the strand and harvest the data
        if i>=minimum_threshold and not ss=='E':
            blocks.append(current_strand)
            block_len.append(len(current_strand))
            current_strand=[]
            i=0
    number_strands=len(blocks)
    mean_len_strands=0
    if len(block_len)>0:
        mean_len_strands=np.array(block_len).mean()
    return number_strands, mean_len_strands

def has_histidine_brace(structure, first_N, max_distance, min_his_his_dist):
    """
    ##Identify a histidine brace by looping over histidine residues
    ##if another histidine is within the max euclidean distance and meets the minimum distance criteria: identify as a *possible* his brace
    ##if both histidines within the possible brace are separated by a minimum of integer "min_his_his_dist" residues return True
    Parameters
    ----------
    structure : structure to interrogate
    first_N : first N residues in which the N-terminal histidine can exist, enables flexibility with signal peptide prediction error
    max_distance : maximum angstrom distance in which a histidine brace is considered a histidine brace 
    min_his_his_dist : minimum residue distance between histidines being considered for a histidine brace, prevents erroneous identifications

    Returns
    -------
    bool: classification LPMO-like = True, non-LPMO-like = False - deprecated, based on random forest classification
    float: NE2 to NE2 angstrom distance between histidine residues in brace
    float: residue distance between histidine residues in brace
    float: NE2 (N term his) to CA on his2 angstrom distance
    float: CA (N term his) to NE2 on his2 angstrom distance
    float: CA (N term his) to CA on his 2 angstrom distance
    """
    global his
    global his1
    model = structure[0]
    residues = list(model.get_residues())
    histidines = [residue for residue in residues if residue.get_resname() in ["HIS", "HSE", "HSD", "HSP"]]
    #create list, this is to handle multiple brace occurances per protein
    brace_data=[]
    ###check for proteins without any His
    if histidines==[]:
         print("No histidines in structure")
         return False, [[0,0,0,0,0,0,0,0]]
    if histidines[0].id[1]>first_N:
         print("First histidine outside of first_N limit")
         return False, [[0,0,0,0,0,0,0,0]]
    ###check if his within first N residues
    if histidines[0].id[1]<=first_N:
        #his1_ne2 = his1['NE2']
        his1 = histidines[0]
    brace_num=1
    ###if more than one his in protein
    if len(histidines)>1:
         for his in histidines[1:]:  # exclude the first histidine from the loop as stored as variable above, only loop subsequent histidines
             #his1_ne2 = his1['NE2']
             #his_ne2 = his['NE2']
             ##calculate the distance between the nitrogen atoms in the histidines (NE2)
             ##subtracting atom objects return euclidean distance in angstrom of objects
             NE2_NE2_distance=calculate_atom_distance(his1, his, 'NE2','NE2')
             ##grab the index (id) of the residue in the protein sequences and calculate distance in residues between them
             his_his_residue_distance=abs(his1.id[1] - his.id[1])
             ###check the number of residues between the histidines to avoid false positives  
             if his_his_residue_distance >= min_his_his_dist:
                 ###check angstrom distance between the histidine residues
                 if NE2_NE2_distance <= max_distance:
                     ##ascertain data for NE2 to CA backbone, for orientation elucidation
                      NE2_1_CA_2_distance=calculate_atom_distance(his1, his, 'NE2','CA')
                      CA_1_NE2_2_distance=calculate_atom_distance(his1, his, 'CA','NE2')
                      CA_1_CA_2_distance=calculate_atom_distance(his1, his, 'CA','CA')
                      ###return True and values for metadata
                      #collate data as [brace number, residue 1 position, residue 2 position, NE2_NE2_distance, his_his_residue_distance, NE2_1_CA_2_distance, CA_1_NE2_2_distance, CA_1_CA_2_distance]
                      brace_data.append([brace_num, his1.id[1], his.id[1], his_his_residue_distance, NE2_NE2_distance, NE2_1_CA_2_distance, CA_1_NE2_2_distance, CA_1_CA_2_distance])
                      brace_num+=1
                      #return True, NE2_NE2_distance, his_his_residue_distance, NE2_1_CA_2_distance, CA_1_NE2_2_distance, CA_1_CA_2_distance, brace_data
    if len(brace_data)>=1:
         return True, brace_data
    #default to False if not more than 1 his 
    #print("Structure fail", structure, brace_data)
    return False, [[0,0,0,0,0,0,0,0]]

def has_beta_strand_core(pdb_file, min_beta_strands, min_beta_length, ss_type):
    """
    Determines the number and length of beta strands

    Parameters
    ----------
    pdb_file : pdb structure to interrogate
    min_beta_strands : number of beta strands to consider (deprecated)
    min_beta_length : minimum length of beta strand to consider (depreacted)

    Returns
    -------
    bool: classification LPMO-like = True, non-LPMO-like = False - deprecated, based on random forest classification
    number_beta_strands : number of beta strands
    mean_length_beta_strands : mean length of beta strands
    beta_strand_count_above_min : number of strands above minimum length
    secondary_structure : seconary structure

    """
    #Use mdtraj to compute secondary structure, if number of strands at the minimum length are identified return True else False
    traj = md.load(pdb_file)
    secondary_structure = md.compute_dssp(traj)[0]  # take the first frame (assuming only 1 frame)
    if ss_type==False:
        #compute COMPLEX secondary structure
         secondary_structure = md.compute_dssp(traj, simplified=False)[0]
         ## NB Loops and irregular elements are represented as empty string - problem - solution =  change these to 'X'
         secondary_structure = ['X' if item == ' ' else item for item in secondary_structure]
    ###get absolute values for the proteins number of strands and the mean length for metadata
    number_beta_strands, mean_length_beta_strands = count_beta_strands(secondary_structure, min_beta_length)
    beta_strand_count_above_min = 0
    current_length = 0
    ##count number of strands over the minimum length - this could be one function within count_beta_strands but for readability keep as is
    for ss in secondary_structure:
        if ss == 'E':  # extended beta strand in DSSP classification see https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html
            current_length += 1
        else:
            if current_length >= min_beta_length:
                beta_strand_count_above_min += 1
            current_length = 0
    #If last residue is also a beta strand
    if current_length >= min_beta_length:
        beta_strand_count_above_min += 1
    if beta_strand_count_above_min >= min_beta_strands:
        return True, number_beta_strands, mean_length_beta_strands, beta_strand_count_above_min, secondary_structure
    return False, number_beta_strands, mean_length_beta_strands, beta_strand_count_above_min, secondary_structure

def output_file(positive_hit_list, outpath):
    """
    Parameters
    ----------
    positive_hit_list : names of files that were classified positively
    outpath : output folder
    
    Returns
    -------
    None.

    """
    ###Write filenames of positive identifications to text file for futher scrutiny
    f=open(os.path.join(outpath, "LYFE_out_LPMO_pdbs.txt"), 'w')
    for file in positive_hit_list:
        f.write(str(file)+'\n')
    f.close()

def exit_error_countdown():
    """
    Exits main function if prerequisites fail to load

    Returns
    -------
    system exit
    """
    for i in range(5):
          print("Exiting in ", 5-i)
          time.sleep(1)
    return sys.exit()

def open_new_file(filename, mode, header, openfilelist, report_secondary_structue):
    """
    Open new file for appending data 

    Parameters
    ----------
    filename : filename to open
    mode : mode for file
    header : header to write
    openfilelist : list to append names of open files to
    report_secondary_structue : bool if seconary structure is to be recorded

    Returns
    -------
    openfile : an open file objects

    """
    # get current time in seconds since epoch
    current_time = time.time()
    # convert the timestamp to a struct_time
    time_struct = time.localtime(current_time)
    formatted_time = time.strftime("%d%m_%H_%M_%S", time_struct)
    newfilename=filename+'_'+formatted_time+'.csv'
    openfile=open(newfilename, mode)
    openfile.write(','.join(header)+'\n')
    openfilelist.append(openfile)
    return openfile

def append_to_file(file, data):
    """
    Write new data to open file

    Parameters
    ----------
    file : open file object
    data : data to be written (list)

    Returns
    -------
    None.

    """
    file.write(','.join([str(x) for x in data])+'\n')
    file.flush()  # ensure the data is written to disk immediately
    print("Updated file")
    return

def close_files(openfilelist):
    """
    Parameters
    ----------
    openfilelist : List of open files to close.

    Returns
    -------
    None.

    """
    for file in openfilelist:
        file.close()

def append_secondary_structure_to_existing_df(row_index, SS_data_list, metadata, metadata_cols, dataframe, maximum_csv_length_to_report):
    """
    Parameters
    ----------
    row_index : index of row
    SS_data_list : secondary structure data as list
    metadata : metadata as list
    metadata_cols : column names for metadata
    dataframe : dataframe for modification
    maximum_csv_length_to_report : maximum number of columns to report

    Returns
    -------
    dataframe : modified dataframe
    """
    # Create a temporary dataframe from the metadata list with the same number of rows as the original dataframe
    fulldata = metadata + list(SS_data_list)
    
    #Check if data is above maximum length limit
    if len(fulldata) >= maximum_csv_length_to_report:
        print("ERROR: Length of data exceeds CSV length limit")   
    newheader = metadata_cols + [f'SS_{i + 1}' for i in range(maximum_csv_length_to_report - len(metadata_cols))]
    missing_length = maximum_csv_length_to_report - len(fulldata)
    # Append 0s to data if shorter than maximum limit
    fulldata += [0] * missing_length
    #print(f"Missing length: {missing_length}, Total length of fulldata: {len(fulldata)}")  
    # Create a temporary DataFrame
    temp_df = pd.DataFrame([fulldata], columns=newheader)
    # Ensure proper data types
    for col in temp_df.columns:
        # change the dtype to a more specific type if necessary
        if temp_df[col].dtype == 'object':
            temp_df[col] = temp_df[col].astype(str)
    
    # Concat the original dataframe and the temporary dataframe
    dataframe = pd.concat([dataframe, temp_df], axis=0, ignore_index=True)
    # replace NaN values with 0
    dataframe = dataframe.fillna(0)
    for col in dataframe.columns:
        if col.startswith('SS_'):  # assuming SS columns should be numeric
            dataframe[col] = dataframe[col].astype(str)
    # Print the modified DataFrame
    return dataframe, newheader

def dataframe_handler(A, i, metadata_cols, metadata, dataframe, SS, maximum_csv_length_to_report):
    """
    Handles mandatory dataframe input OG_columns if secondary structure is supplied or not 
    Parameters
    ----------
    A : bool secondardy_structure_data supplied TRUE or FALE
    i : index of file (int)
    metadata_cols : column names for 3D metadata for structure (not secondary)
    metadata : 3D metadata for structure (not secondary)
    dataframe : dataframe for modification / updating
    SS : secondary structure data

    Returns
    -------
    dataframe : updated dataframe
    """
    if A:
        dataframe, header=append_secondary_structure_to_existing_df(i, SS, metadata, metadata_cols, dataframe, maximum_csv_length_to_report)
    else:
        dataframe=dataframe.loc[i]=metadata
        header=metadata_cols
    return dataframe, list(dataframe.iloc[-1]), header

def instantiate_progress_bar(total_files):
    """
    Parameters
    ----------
    total_files : total number of files to parse (int)
    Returns
    -------
    bar : tqdm bar object
    """
    bar = trange(total_files, desc=f"Processing 0/{total_files}", leave=True, position=0)
    return bar

def update_progress_bar(bar, total_files, i, pos): 
    """
    Parameters
    ----------
    bar : tqdm bar object
    total_files : total number of files to parse (int)
    i : index of currently completed file (int)
    pos : number of positively classified proteins (int)
    
    Returns
    ----------
    - None
    """
    bar.set_description(f"Processing {i}/{total_files}")
    bar.set_postfix_str(f"Braces detected: {pos}")
    bar.update(i)
    bar.refresh() # to show immediately the update

def load_model_with_encoders(model_path, encoder_path, factormap_path):
    """
    Parameters
    ----------
    model_path : path to model h5
    encoder_path: path to encoder pkl
    Returns
    -------
    NNmodel : loaded model object
    label_encoder : loaded encoder array
    """
    print("Loading deep learning model")
    NNmodel=load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    factor_map = joblib.load(factormap_path)
    print("Loaded deep learning model")
    return NNmodel, label_encoder, factor_map

def plot_probability_bars(probabilities, label_encoder, ID, outpath, classification):
     """
     Plots probability across all classifications in the model
     Parameters
     ----------
     probabilities : predicted probabilities of model
     label_encoder : label encoder for model 
     ID : protein name
     outpath : destination for output
 
     Returns
     -------
     None.
 
     """
     # Get the classes from label_encoder
     classes = label_encoder.classes_
     # Define a rainbow colormap
     colormap = plt.cm.rainbow
     # Plotting
     fig, ax = plt.subplots()
     ax.bar(classes, probabilities[0], linewidth=2, edgecolor='black', color=colormap(probabilities[0]))
     plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
     plt.xlabel('Classes')
     plt.ylabel('Predicted Probabilities')
     plt.title('Predicted Probabilities for Each Class')
     # Adding colorbar to show the mapping of colors to probabilities
     norm = plt.Normalize(0, 1.025)
     sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
     sm.set_array([])
     cbar = plt.colorbar(sm, ax=ax)
     cbar.set_label('Probability')
     ax.set_ylim([0,1])
     fullpath=os.path.join(outpath, "Figures", ID[:-4]+ ".png")
     fig.savefig(fullpath, dpi=600)
     if not classification=='OTHER':
         fig.savefig(os.path.join(outpath, "Figures_classified", ID[:-4]+ ".png"), dpi=600)
     if classification=='OTHER':
         fig.savefig(os.path.join(outpath, "Figures_unclassified", ID[:-4]+ ".png"), dpi=600)
     plt.close()
     return

def get_classification(probabilities, label_encoder, threshold):
    """
    Use imported deep learning model to classify data and retrieve encoded classification from label encoder

    Parameters
    ----------
    probabilities : predicted probabilities
    label_encoder : the label encoder to map class labels.
    threshold : threshold required to classify a protein

    Returns
    -------
    classification: classification for the protein

    """
    if np.max(probabilities) >= threshold:
        predicted_class_index = np.argmax(probabilities)
        classification = label_encoder.classes_[predicted_class_index]
    else:
        classification = "Uncertain"
    return classification 

def check_output_folder_exists(outpath, folder):
    """
    Check if output folders exist, if not create them

    Parameters
    ----------
    outpath : path to output folder

    Returns
    -------
    None.

    """
    os.makedirs(os.path.join(outpath, folder), exist_ok=True)

def classify_with_threshold(model, column, threshold, label_encoder, factor_map, deep_learning_hardstop,  plots, outpath, output):
    """
    Classifies a column using the given model, assesses the threshold level, and returns the classification.
    Parameters:
    - model: the classification model.
    - column: the dataframe column to be classified.
    - threshold: the threshold level for classification.
    - label_encoder_dict: the label encoder to map class labels.
    - hardstop: the length of the model, trims or expands input data to fit the model 
    - plots: bool - plots barplot of all probabilities
    Returns:
    - classified_labels: The classification result based on the threshold level and label encoder dictionary
    """
    structureID=os.path.split(column['Structure'])[-1]
    #remove metadata columns
    slimmed_column=column.drop(labels=[x for x in ['Unnamed: 0', 'Index', 'Structure', 'classification', 'ID'] if x in column.index])
    #factorise using the training factor_map set and cut the data to the hardstop if > hardstop
    factorised_column=slimmed_column.apply(lambda x: factor_map.get(x, x))[:deep_learning_hardstop].astype('float32')    
    if len(factorised_column) < deep_learning_hardstop:
        # pad the column with zeros to make its length equal to deep_learning_hardstop
        #factorised_column = list(factorised_column).append(pd.Series([0] * (deep_learning_hardstop - len(factorised_column))))
        factorised_column = pd.concat([factorised_column, pd.Series([0] * (deep_learning_hardstop - len(factorised_column)))])
    #reshape data for prediction
    input_data = np.array(factorised_column).reshape(1, -1)
    # predict using the reshaped input data
    input_data=np.asarray(input_data).astype('float32')
    predicted_probs = model.predict(input_data)
    classification=get_classification(predicted_probs,label_encoder, threshold)
    if output:
         if plots:
             plot_probability_bars(predicted_probs, label_encoder, structureID, outpath, classification)  
    # Determine the class labels based on the threshold level
    return classification

def detect_model_files(model_dir):
    """
    Discover the .h5 model, label encoder, and factor map 
    inside a specific directory
    """
    if not os.path.exists(model_dir):
        sys.exit(f"Error: The model directory '{model_dir}' does not exist.")

    h5_path = None
    le_path = None
    fm_path = None

    print(f"Scanning {model_dir} for model files...")

    for filename in os.listdir(model_dir):
        full_path = os.path.join(model_dir, filename)
        
        if filename.endswith(".h5"):
            h5_path = full_path
        elif filename.endswith("_label_encoder.pkl"):
            le_path = full_path
        elif filename.endswith("factor_map.pkl"):
            fm_path = full_path

    # Error handling if files are missing
    missing = []
    if h5_path is None: missing.append(".h5 model file")
    if le_path is None: missing.append("_label_encoder.pkl")
    if fm_path is None: missing.append("_factor_map.pkl")

    if missing:
        sys.exit(f"Error: Could not find the following in '{model_dir}': {', '.join(missing)}")
    print(h5_path, le_path, fm_path)
    return h5_path, le_path, fm_path


def main(folder, outpath, file_range,                                        
         first_N=30, max_distance=11, min_his_his_dist=5,                     
         min_beta_strands=1, min_beta_length=3, output=True, 
         report_secondary_structue=True, deep_learning=True, 
         threshold=0.5, graphical_output=True, ss_type=True, 
         fileprefix="CONFERS_", model_dir="FedeAI_1pt6AE"):    
    NNmodel = None 
    label_encoder = None
    factor_map = None
    global main_metadata
    global openfiles
    hardstop = 1997
    maximum_csv_length_to_report = 1997
    openfiles = []
    header = ['Index', 'Structure', 'Brace_ID', 'Res1pos', 'Res2pos', 'his_his_res_dist', 
              'NE2_NE2_dist', 'NE2_CA_dist', 'CA_NE2_dist', 'CA_CA_dist', 'Num_beta_strands', 
              'Mean_beta_strand_len', 'Beta_strand_count_above_min']
    
    check_output_folder_exists(outpath, "Figures")
    main_metadata = pd.DataFrame(columns=header)     
    pdb_folder = folder
    pdb_files = [os.path.join(pdb_folder, file) for file in os.listdir(pdb_folder) if file.endswith(".pdb")]
    tot_files=len(pdb_files)
    print("Total PDB files in target folder parse: ", str(len(pdb_files)))
    # Handle file range if provided
    if file_range:
        print("Custom file range detected")
        try:
            start, end = map(int, file_range.split('-'))
            pdb_files = pdb_files[start:end]  # Filter the files based on the range
            print("###\n" + f"Parsing files from index {start} to {end} out of {len(pdb_files)} total files.\n###")
            tot_files=len(pdb_files)
        except ValueError:
            print("Invalid file range specified. Please use the format start-end (e.g., 0-20000).")
            return
    else:
        print("Parsing all files from ", str(folder))

    print("Total PDB files to parse: ", str(tot_files))
    
    parser = PDBParser(QUIET=True)
    fileprefix = fileprefix + '_' + os.path.split(folder)[-1]
    valid_pdb_files = []
    count = 0
    fails = 0
    i = 0
    DLi = 0
    LPMO_metadata = []
    if deep_learning:
        print(f"Attempting to load model from: {model_dir}")
        try: 
             # Construct the path: e.g., "Model_data/FedeAI_1pt6AE"
             base_model_dir = "Model_data"
             target_model_dir = os.path.join(base_model_dir, model_dir)
             h5_path, le_path, fm_path = detect_model_files(target_model_dir)
             #Load models
             NNmodel, label_encoder, factor_map = load_model_with_encoders(h5_path, le_path, fm_path)               
             DLcolumns=header[:1]+['Classification']+header[1:]
             LPMO_metadata=pd.DataFrame(columns=DLcolumns)      
             check_output_folder_exists(outpath, "Figures_classified")
             check_output_folder_exists(outpath, "Figures_unclassified")
             ### INPUT SHAPE DETECTION ###
             print("Detecting input shape...")
             shape_detected = False          
             try:
                 #attempt 1: get_config() (handles Keras 2 and Keras 3 different loadin methods)
                 config = NNmodel.get_config()
                 
                 if 'layers' in config:
                     first_layer = config['layers'][0]
                     layer_config = first_layer.get('config', {})
                     
                     # check 1: Standard Keras (batch_input_shape)
                     if 'batch_input_shape' in layer_config:
                         shape_tuple = layer_config['batch_input_shape']
                     # check 2: Newer Keras (batch_shape)
                     elif 'batch_shape' in layer_config:
                         shape_tuple = layer_config['batch_shape']
                     else:
                         shape_tuple = None
                     
                     # Extract from 'batch_shape' tuple (~Nonw, NNN)
                     if shape_tuple and len(shape_tuple) > 1:
                         input_dimension = shape_tuple[1]
                         shape_detected = True
                         print(f"Shape detected via config key: {shape_tuple}")

                 # Strategy 2: Tensor (fallback if the above do not work)
                 if not shape_detected:
                     input_obj = NNmodel.layers[0].input
                     if isinstance(input_obj, list):
                         input_dimension = input_obj[0].shape[1]
                     else:
                         input_dimension = input_obj.shape[1]
                     shape_detected = True
                     print("Shape detected via input tensor")

             except Exception as shape_error:
                 print(f"Warning: Auto-detection of input shape failed ({shape_error}).")
                 print(f"Defaulting to hardcoded value: {input_dimension}")

             print(f"Final Input Dimension set to: {input_dimension}")
             hardstop = input_dimension

        except Exception as e:
             traceback.print_exc()
             print(f"\nCRITICAL ERROR: Could not load model. Deep learning disabled.\nError: {e}")
             deep_learning = False    
    LYFE_file = None
    for pdb_file in pdb_files:
        bar = instantiate_progress_bar(len(pdb_files))
        count += 1
        try:
            structure = parser.get_structure("structure", pdb_file)
            his_brace_bool, brace_data = has_histidine_brace(structure, first_N, max_distance, min_his_his_dist)
            beta_strand_bool, number_beta_strand, mean_len_beta_strands, beta_strand_count_above_min, secondary_structure = has_beta_strand_core(pdb_file, min_beta_strands, min_beta_length, ss_type)
            
            ## Handle hard metadata and multiple his braces
            for brace in list(range(0, len(brace_data))):
                metadata = [str(i), str(pdb_file), brace_data[brace][0], brace_data[brace][1], brace_data[brace][2],
                            brace_data[brace][3], brace_data[brace][4], brace_data[brace][5], brace_data[brace][6],
                            brace_data[brace][7], str(number_beta_strand), str(mean_len_beta_strands), 
                            str(beta_strand_count_above_min)]
                
                main_metadata, updated_metadata, updated_header = dataframe_handler(report_secondary_structue, i, header, metadata, main_metadata, secondary_structure, maximum_csv_length_to_report)        
                
                if output and LYFE_file is None:
                    if i == 0 and brace == 0:
                        # Write header
                        LYFE_file = open_new_file(os.path.join(outpath, fileprefix + "LYFE_output.csv"), 'a', updated_header, openfiles, report_secondary_structue)

                append_to_file(LYFE_file, updated_metadata)
                
                if deep_learning:
                    classification = classify_with_threshold(NNmodel, main_metadata.iloc[-1], threshold, label_encoder, factor_map, hardstop, graphical_output, outpath, output)
                    if classification != 'OTHER':
                        print('\n', classification)
                        valid_pdb_files.append(pdb_file)
                        DLmetadata = metadata[:1] + [classification] + metadata[1:]
                        LPMO_metadata, updated_DLmetadata, updated_header = dataframe_handler(report_secondary_structue, i, DLcolumns, DLmetadata, LPMO_metadata, secondary_structure, maximum_csv_length_to_report)

                        if output:
                            if DLi == 0:
                                LYFE_DL_file = open_new_file(os.path.join(outpath, fileprefix + "LYFE_classifications.csv"), 'a', updated_header, openfiles, report_secondary_structue)
                            append_to_file(LYFE_DL_file, updated_DLmetadata)
                        
                        DLi += 1
                        print(DLi)
                        break
            
            i += 1
            update_progress_bar(bar, tot_files, i, str(len(LPMO_metadata)))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('\nFAILURE: ', str(pdb_file), ' error:', e)
            print(exc_type, fname, exc_tb.tb_lineno)
            fails += 1
            print('\n')           

    print("Valid PDB files:")
    ### Write filenames of his brace containing proteins to txt file
    output_file(valid_pdb_files, outpath)
    close_files(openfiles)
    return valid_pdb_files

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run protein analysis.')

    # Add arguments for each of the parameters
    parser.add_argument('--pdb_folder', type=str, required=True, help='Path to the folder containing PDB files')
    parser.add_argument('--first_N', type=int, default=30, help='First N residues to search for N terminal histidine')
    parser.add_argument('--max_distance', type=int, default=11, help='Max histidine distance')
    parser.add_argument('--min_his_his_dist', type=int, default=5, help='Min his-his distance')
    parser.add_argument('--min_beta_strands', type=int, default=1, help='Min beta strands')
    parser.add_argument('--min_beta_length', type=int, default=3, help='Min beta length')
    parser.add_argument('--output', type=bool, default=True, help='Generate output')
    parser.add_argument('--folder', type=str, default="", help='Output folder')
    parser.add_argument('--outpath', type=str, default="FINAL_PAPER_LYFE_output_", help='Output path for filenames')
    parser.add_argument('--report_secondary_structue', type=bool, default=True, help='Report secondary structure')
    parser.add_argument('--deep_learning', type=bool, default=True, help='Use deep learning')
    parser.add_argument('--classification_threshold', type=float, default=0.7, help='Classification threshold')
    parser.add_argument('--graphical_output', type=bool, default=True, help='Generate graphical output')
    parser.add_argument('--sstype', type=bool, default=True, help='Simple secondary structure (True) or complex (False)')
    parser.add_argument('--file_range', type=str, default=None, help='Range of files to parse (e.g., 0-20000)')
    parser.add_argument('--model_name', type=str, default="FedeAI_1pt6AE", help='Name of the subfolder in Model_data containing the model files')

    # parse the arguments
    args = parser.parse_args()
    # timing the execution
    tic = time.perf_counter()
    # call the main function with the parsed arguments
    valid = main(
        #mandatory args
        folder=args.pdb_folder,
        outpath=args.outpath,
        file_range=args.file_range,

        #optional args
        first_N=args.first_N,
        max_distance=args.max_distance,
        min_his_his_dist=args.min_his_his_dist,
        min_beta_strands=args.min_beta_strands,
        min_beta_length=args.min_beta_length,
        output=args.output,
        report_secondary_structue=args.report_secondary_structue,
        deep_learning=args.deep_learning,
        threshold=args.classification_threshold,
        graphical_output=args.graphical_output,
        ss_type=args.sstype,
        model_dir=args.model_name
        )
    toc = time.perf_counter()
    print(f"Performed searches in {toc - tic:0.2f} seconds")
    print("Processing complete")

