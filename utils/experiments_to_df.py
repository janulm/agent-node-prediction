# written by Jannek Ulm 5.1.2023
import pandas as pd

def _namespace_line_to_args(line,args):
    line = line[:-2]
    line = line.split("Namespace(")[1]
    things = line.split(",")
    things_dict = {}
    for i in range(len(things)):
        things[i] = things[i].replace(' ','').split("=")
        try:
            fval = float(things[i][1])
        except ValueError:
            # handle strings, contains strategies, dataset, activation function etc.
            fval = things[i][1]
        things_dict[things[i][0]] = fval
    #print(things_dict)
    
    res = {}
    for arg in args:
        key = arg[0]
        name = arg[1] if len(arg) == 2 else key
        if key in things_dict:
            val = things_dict[key]
            if type(val) == float:
                res[name] = things_dict[key]
            elif type(val) == str:
                res[name] = things_dict[key].split(".")[1].split(":")[0]
            else:
                raise Exception("Unknown type of value")
        else:
            raise Exception("Key not found in Namespace line")
    # args is a list of strings
    # line is a string
    # returns a list of strings/floats
    #print(res)
    return res

def _result_line_to_args(line,args):
    # universal for all result lines
    line = line[:-1]

    if "test_acc" in line and len(args) == 1:
        # only test acc line
        line = line.split("test_acc: ")[1]
        acc = float(line)
        res = {}
        res[args[0][1]] = acc
        return res
    elif "test_loss" in line and "spear coeff" in line and len(args) == 2:
        # test loss and spear coeff line
       line = line.split(", device='cuda:0') spear coeff:  tensor(")

       loss = float(line[0].split('(')[1])
       spear = float(line[1].split(',')[0])
       res = {}
       res[args[1][1]] = loss
       res[args[0][1]] = spear
       return res
    else:
        # something wrong line
        raise Exception("Unknown result line")


def filter_df(df,filter_list):
    cols = df.columns
    last_col = cols[-1]
    df = df.sort_values(last_col, ascending=False, inplace=False)
    
    # drop duplicates for keys
    keepList = []
    for f in filter_list:
        keepList.append(f)
    df = df.drop_duplicates(subset=keepList,inplace=False)
    
    # resort by desired metric (low to high) such that high gets drawn on top
    df = df.sort_values(last_col, ascending=True, inplace=False)
    return df 


def exp_run_to_df(file_path,name_space_args,res_args,filter_flag=True):
    # file_path is a string describing the path to the file
    # name_space_args is a list of strings describing the arguments in Namespace lines
    # res_args is a list of strings describing the arguments in result lines such as test_acc or spear coeff
    # if filter_flag will filter for the best unique runs (defined by res_args[0])
    d = []
    file = open(file_path, "r")
    lines = file.readlines()
    i = 0
    while i < len(lines):
        if "Namespace" in lines[i] and i+1 < len(lines) and res_args[0][0] in lines[i+1]:
            args = _namespace_line_to_args(lines[i],name_space_args)
            res = _result_line_to_args(lines[i+1],res_args)
            for k in res:
                args[k] = res[k]
            d.append(args)
            i += 2
        else:
            i += 1

    df = pd.DataFrame(d)
    
    df.sort_values(res_args[0][1], ascending=False, inplace=True)
    if filter_flag:
        # drop duplicates for keys
        df.drop_duplicates(subset=df.columns.difference([res_args[0][1]]),inplace=True)
    
    # resort by desired metric (low to high) such that high gets drawn on top
    df.sort_values(res_args[0][1], ascending=True, inplace=True)
    
    return df