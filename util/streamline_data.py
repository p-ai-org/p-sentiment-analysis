from compile_spreadsheets import compile_sheets
from clean_csv import clean_csv
from vectorize_data import save_vectored_data, save_averaged_vectored_data

def streamline_data(a_training_sheet_fname, b_training_sheet_fname, averaged, fname):
    # Compile google sheets
    compiled = compile_sheets(a_training_sheet_fname, b_training_sheet_fname)
    print(' [Compiled sheets]')
    # Clean
    cleaned = clean_csv(compiled)
    print(' [Cleaned data]')
    # Apply vectorization
    if (averaged):
        save_averaged_vectored_data(df = cleaned, model_name = 'model_5', fname = fname, vector_size = 100)
    else:
        save_vectored_data(df = cleaned, model_name = 'model_5', fname = fname, vector_size = 100)
    print(' [Vectored data]')
    print(' [Done]')

streamline_data(    a_training_sheet_fname = 'a_training_sheet', 
                    b_training_sheet_fname = 'b_training_sheet', 
                    averaged = True, 
                    fname = 'streamlined_data')