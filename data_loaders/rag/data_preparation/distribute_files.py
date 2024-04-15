import os
from collections import defaultdict
import pathspec

def find_first_gitignore(directory):
    """Searches for the first .gitignore file in the directory tree."""
    for root, dirs, files in os.walk(directory):
        if '.gitignore' in files:
            return os.path.join(root, '.gitignore')
    return None

def read_gitignore_rules(gitignore_path):
    """Reads .gitignore rules from the provided path and returns clean patterns."""
    patterns = []
    try:
        with open(gitignore_path, 'r') as gitignore_file:
            for line in gitignore_file:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    patterns.append(stripped_line)
        return patterns
    except FileNotFoundError:
        return []

def get_directory_sizes_and_files(startpath, spec=None, include_spec=None):
    if not os.path.isdir(startpath):
        raise ValueError(f"The startpath must be a valid directory. Given: {startpath}")
    
    dir_info = defaultdict(lambda: {'size': 0, 'files': []})
    
    for root, dirs, files in os.walk(startpath, topdown=True):
        rel_root = os.path.relpath(root, startpath)
        if spec:
            # Exclude files and dirs not matching include patterns (if include_spec is specified) or matching exclude patterns
            dirs[:] = [d for d in dirs if not spec.match_file(os.path.join(rel_root, d) + '/')] # Exclude directories
            files = [f for f in files if (include_spec.match_file(os.path.join(rel_root, f)) if include_spec else not spec.match_file(os.path.join(rel_root, f)))]
        
        for name in files:
            file_path = os.path.join(root, name)
            file_size = os.path.getsize(file_path)
            dir_info[root]['size'] += file_size
            dir_info[root]['files'].append(file_path)
            
    return dir_info


def distribute_into_buckets(startpath, n_buckets=10, use_gitignore=True, exclude_patterns=[], include_patterns=None):
    gitignore_path = find_first_gitignore(startpath) if use_gitignore else None
    gitignore_patterns = read_gitignore_rules(gitignore_path) if gitignore_path else []
    combined_patterns = gitignore_patterns + exclude_patterns
    spec = pathspec.PathSpec.from_lines('gitwildmatch', combined_patterns) if combined_patterns else None
    include_spec = pathspec.PathSpec.from_lines('gitwildmatch', include_patterns) if include_patterns else None
    
    dir_info = get_directory_sizes_and_files(startpath, spec, include_spec=include_spec)
    sorted_dirs = sorted(dir_info.items(), key=lambda x: x[1]['size'], reverse=True)
    
    buckets = [[] for _ in range(n_buckets)]
    bucket_sizes = [0] * n_buckets
    
    for directory, info in sorted_dirs:
        min_bucket_index = bucket_sizes.index(min(bucket_sizes))
        bucket = {
            'path': directory, 
            'size': info['size'], 
            'files': info['files']
        }
        buckets[min_bucket_index].append(bucket)
        bucket_sizes[min_bucket_index] += info['size']

    return buckets


def extract_file_paths(data):
    all_file_paths = []
    # Iterate through each dictionary in the list
    for item in data:
        # Extract the list of files for the current dictionary
        files = item.get('files', [])
        # Add the files to the overall list of file paths
        all_file_paths.extend(files)
    return all_file_paths


@data_loader
def load_data(local_dir, *args, **kwargs):
    sample = kwargs.get('sample', 2)

    exclude_patterns = [
        '*.txt',
    ]
    buckets = distribute_into_buckets(
        local_dir, 
        n_buckets=40, 
        include_patterns=['*.mdx'],
        use_gitignore=True, 
        exclude_patterns=exclude_patterns,
    )

    verbose = kwargs.get('verbose', False)
    if verbose:
        for i, bucket in enumerate(buckets):
            print(f"Bucket {i+1}:")
            for item in bucket:
                print(f"  {item['path']} - {item['size']} bytes")
            print("------")

    arrs = []
    for bucket in buckets[:sample]:
        files = extract_file_paths(bucket)
        arrs.append(files[:sample + 1])

    return [
        arrs,
    ]