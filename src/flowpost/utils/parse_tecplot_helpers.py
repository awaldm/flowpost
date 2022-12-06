'''
helper functions for parsing Tecplot-formatted ASCII data
'''
import numpy as np
import pandas as pd
# identify column variable positions
# inputs: - variables: the list of strings containing the parsed vars

def find_var_positions(variables, *args):
    pos = {}
    for pos_count, searchvar in enumerate(args):
        pos[searchvar] = int(variables.index(searchvar))
    print(type(variables))
    return pos

# old, hardcoded function
def find_var_forces_old(variables):
    print(type(variables))
    time_col = int(variables.index("thistime"))

    CL_col = int(variables.index("C-lift"))
    CD_col = int(variables.index("C-drag"))
    CM_col = int(variables.index("C-my"))
    return time_col, CL_col, CD_col, CM_col

def filter_inner_iterations(in_data,tcol):
    """Checks if there is inner iteration output in the file and filters this.
    
    This checks for multiple lines with the same time and keeps only the last row. The idea is that this
    does not care how many variables (i.e columns) there are.
    
    Input:
        data: the original data as numpy matrix, shape (numrows, num_variables)
        tcol: integer number of the time variable column.
    Output:
        filtered: filtered data as python list, call np.asarray(filtered) on this if desired

    """
    if isinstance(in_data, pd.DataFrame):
        if not isinstance(tcol, int):
            tcol = in_data.columns.get_loc(tcol)
        columns = in_data.columns
        data = in_data.to_numpy()

    # first time
    current_time = data[0,tcol]
    # output array
    filtered = []
    # overall number of rows in data
    numrows = len(data[:,tcol])

    # start from second line (otherwise row-1 wouldnt exist)
    for row in range(1,numrows):
        # find new time step, append the previous line
        if (data[row,tcol] > current_time):
            #print('last iteration of timestep '+str(data[row,tcol]))
            filtered.append(data[row-1,:])
            current_time = data[row,tcol]
    # append last line
    filtered.append(data[-1,:])
    if isinstance(in_data, pd.DataFrame):
        return pd.DataFrame(filtered, columns = columns)
    else:
        return np.asarray(filtered)

