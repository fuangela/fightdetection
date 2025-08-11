from processdata import DataProcessor

dp = DataProcessor()
print(hasattr(dp, 'prepare_training_data'))

import processdata
print(processdata.__file__)
