def parse_param_file(filename):
    """ 
    Parse the param file the user has inputted and fill a dictionary with
    all the required values.
    """
    params = {}

    with open(filename, "r") as infile:
        lines = infile.readlines()
        for line in lines:
            if line[0] == "#" or len(line) == 0: # Ignore comments and blank lines
                continue
            # Split the line at the equals sign, ignoring any trailing comments
            if not line.find("="):
                print("Invalid parameter line, no value found.")
                continue
            if line.find("#"):
                split_line = line.split("#")[0]
            split_line = split_line.split("=")
            param, value = split_line[0].strip(), split_line[1].strip()

            if param == "FILENAME_PREFIX":
                # Filename prefix is used for input data and output data
                params["FILENAME_PREFIX"] = value
            elif param == "DIM":
                params["DIM"] = int(value)
            elif param == "BASES":
                if value == "all":
                    params["BASES"] = value
                else:
                    params["BASES"] = [int(h) for h in value.split(",")]
            elif param == "N_TRIALS":
                params["N_TRIALS"] = int(value)
            elif param == "N_WORKERS":
                params["N_WORKERS"] = int(value)
            elif param == "PERCENT_TEST":
                params["PERCENT_TEST"] = float(value)
            elif param == "PERCENT_VAL":
                params["PERCENT_VAL"] = float(value)
            elif param == "HIDDEN_LAYER_SIZES":
                params["HIDDEN_LAYER_SIZES"] = [int(h) for h in value.split(",")]

    params["DATA_IN_FILE"] = params["FILENAME_PREFIX"] + "_d" + str(params["DIM"]) + "_b" + str(params["BASES"]) + "_in.csv"
    params["DATA_OUT_FILE"] = params["FILENAME_PREFIX"] + "_d" + str(params["DIM"]) + "_b" + str(params["BASES"]) + "_out.csv"
    params["LOG_FILE"] = params["FILENAME_PREFIX"] + "_d" + str(params["DIM"]) + "_b" + str(params["BASES"]) + ".log"

    return params

