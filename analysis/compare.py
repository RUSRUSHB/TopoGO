import csv

def load_csv(filename):
    data = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            identifier = row[0]
            data[identifier] = row[1]
    return data

def save_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for identifier, value in data.items():
            writer.writerow([identifier, value])

def compare_and_update(source_file, target_file):
    source_data = load_csv(source_file)
    target_data = load_csv(target_file)

    # Find identifiers in source that are not in target and remove them
    identifiers_to_remove = [identifier for identifier in source_data if identifier not in target_data]
    for identifier in identifiers_to_remove:
        del source_data[identifier]

    save_csv(source_file, source_data)

# 根据你的文件名执行操作
source_file = 'Alex.csv'
target_file = 'Terminal_oneline.csv'
compare_and_update(source_file, target_file)
