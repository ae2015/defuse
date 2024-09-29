import os
import json
import matplotlib.pyplot as plt

def count_json_files(base_dir):
    topic_counts = {}
    
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            topic_dir = os.path.join(root, dir)
            json_files = [f for f in os.listdir(topic_dir) if f.endswith('.json')]
            topic_counts[dir] = len(json_files)
    
    return topic_counts

def plot_topic_counts(topic_counts):
    topics = list(topic_counts.keys())
    counts = list(topic_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(topics, counts, color='skyblue')
    plt.xlabel('Topic')
    plt.ylabel('Number of News Articles')
    plt.title('Number of News Articles per Topic')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('data/imgs/topic_counts.png')

if __name__ == "__main__":
    base_dir = '/WAVE/users2/unix/zpeng/proj/defuse/data/raw/News1k2024'
    topic_counts = count_json_files(base_dir)
    plot_topic_counts(topic_counts)