import json
import os
import pathlib

def update_1_1(index):
    return index

store = os.path.join(pathlib.Path(__file__).parent.absolute(), "store")

symbols = os.listdir(store)

for symbol in symbols:
    symbol_dir = os.path.join(store, symbol)
    if(os.path.isdir(symbol_dir)):
        print (symbol)
        
        timeframes = os.listdir(symbol_dir)

        for timeframe in timeframes:
            time_dir = os.path.join(symbol_dir, timeframe)
            index_path = os.path.join(time_dir, "index.json")
            if (os.path.exists(index_path)):
                with open(index_path, 'r') as file:
                    index = json.loads(file.read())

                    total = 0
                    for i in range(len(index['index'])):
                        with open(os.path.join(time_dir, str(index['index'][i][0]) + ".json"), 'r') as chunk_file:
                            chunk = json.loads(chunk_file.read())
                            total = len(chunk)
                        index['index'][i].append(total)

                with open(index_path, 'w') as file:
                    file.write(json.dumps(index))

                print("    /" + timeframe + " \tDone")
            else:
                print("    /" + timeframe + " \tNo Index")

# read all data:
