import json
import numpy as np

def range_avg(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Extract the range keys from the first dataset
    first_key = next(iter(data))
    range_keys = list(data[first_key].keys())

    # Initialize a dictionary to store sum and count for each range dynamically
    range_stats = {key: {"sum": 0, "count": 0} for key in range_keys}
    # Traverse through the data and update the sums and counts
    for key, value in data.items():
        for range_key, performance in value.items():
            if not np.isnan(performance):
                range_stats[range_key]["sum"] += performance
                range_stats[range_key]["count"] += 1

    # Calculate the average performance for each range
    average_performance = {range_key: (stats["sum"] / stats["count"]) if stats["count"] > 0 else np.nan
                        for range_key, stats in range_stats.items()}

    # Add the average performance results to the data
    data['average_performance'] = average_performance

    # Save the updated data back to the JSON file
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

    # Print the average performance for each range
    print("Path:", json_path)
    print("Average performance for each range:")
    for range_key, avg_perf in average_performance.items():
        print(f"{range_key}: {avg_perf:.2f}")


if __name__ == '__main__':
    json_path = "/home/zikaixiao/zikaixiao/LongLoRA-main/LongBench/pred_e/tinyllama/result.json"
    range_avg(json_path)