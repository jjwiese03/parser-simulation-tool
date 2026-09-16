import random
import json

# Synonyms
pos_keys = ['x', 'pos', 'position', 'positions']
size_keys = ['width', 'thickness', 'size']
dist_keys = ['distance', 'gap', 'distances']
eps_keys = ['epsilon', 'eps', 'eps_r', 'epsilon_r', 'permittivity', 'dielectric_constant']

def rand_float(a=0.1, b=300.0, decimals=2):
    return round(random.uniform(a, b), decimals)

def rand_int_list(min_len=5, max_len=20):
    length = random.randint(min_len, max_len)
    return [random.randint(1, 3) for _ in range(length)]

def generate_data_string(n_entries=None):
    if n_entries is None:
        n_entries = random.randint(4, 12)
    
    fmt = random.choice([
        'list_of_dicts',
        'csv',
        'key_value_lines',
        'json_dict',
        'python_lists',
        'semicolon_kv',
        'layers_list',
        'mixed_pos_size',
        'dict_with_lists',
        'pipe_separated',
        'equals_lines',
        'json_string',
        'compact_dicts',
        'mixed_attrs'
    ])
    
    positions = sorted([rand_float(5, 300) for _ in range(n_entries)])
    sizes = [rand_float(0.3, 10) for _ in range(n_entries)]
    distances = [rand_float(1, 50) for _ in range(n_entries)]
    epsilons = [rand_float(1.5, 6.0, 1) for _ in range(n_entries)]
    
    # Decide which attributes to include
    use_pos = True
    use_size = random.random() > 0.1
    use_dist = random.random() > 0.5
    use_eps = random.random() > 0.4
    
    pos_key = random.choice(pos_keys)
    size_key = random.choice(size_keys)
    dist_key = random.choice(dist_keys)
    eps_key = random.choice(eps_keys)
    
    if fmt == 'list_of_dicts':
        items = []
        for i in range(n_entries):
            d = {}
            # random order of keys
            attrs = []
            if use_pos:
                attrs.append((pos_key if random.random() > 0.3 else random.choice(pos_keys), positions[i]))
            if use_size and random.random() > 0.2:
                attrs.append((size_key if random.random() > 0.3 else random.choice(size_keys), sizes[i]))
            if use_dist and random.random() > 0.4:
                attrs.append((dist_key if random.random() > 0.3 else random.choice(dist_keys), distances[i]))
            if use_eps and random.random() > 0.5:
                attrs.append((eps_key if random.random() > 0.3 else random.choice(eps_keys), epsilons[i]))
            if not attrs:
                attrs = [(pos_key, positions[i]), (size_key, sizes[i])]
            random.shuffle(attrs)
            d = dict(attrs)
            items.append(d)
        # represent as string
        if random.random() > 0.5:
            s = str(items)
        else:
            s = json.dumps(items)
        return s
    
    elif fmt == 'csv':
        headers = []
        if use_pos:
            headers.append(pos_key)
        if use_size:
            headers.append(size_key)
        if use_dist and random.random() > 0.5:
            headers.append(dist_key)
        if use_eps and random.random() > 0.5:
            headers.append(eps_key)
        if len(headers) < 2:
            headers = [pos_key, size_key]
        lines = [','.join(headers)]
        for i in range(n_entries):
            row = []
            for h in headers:
                if h in pos_keys:
                    row.append(str(positions[i]))
                elif h in size_keys:
                    row.append(str(sizes[i]))
                elif h in dist_keys:
                    row.append(str(distances[i]))
                elif h in eps_keys:
                    row.append(str(epsilons[i]))
            lines.append(','.join(row))
        return '\n'.join(lines)
    
    elif fmt == 'key_value_lines':
        lines = []
        for i in range(n_entries):
            parts = []
            if use_pos:
                parts.append(f"{pos_key}: {positions[i]}")
            if use_size and random.random() > 0.2:
                parts.append(f"{size_key}: {sizes[i]}")
            if use_dist and random.random() > 0.5:
                parts.append(f"{dist_key}: {distances[i]}")
            if use_eps and random.random() > 0.5:
                parts.append(f"{eps_key}: {epsilons[i]}")
            if not parts:
                parts = [f"{pos_key}: {positions[i]}", f"{size_key}: {sizes[i]}"]
            sep = random.choice([' | ', '; ', ', ', '  '])
            lines.append(sep.join(parts))
        return '\n'.join(lines)
    
    elif fmt == 'json_dict':
        d = {}
        if use_pos:
            d[pos_key if pos_key.endswith('s') else pos_key + 's' if random.random()>0.5 else pos_key] = positions
        if use_size:
            sk = size_key if size_key.endswith('s') else size_key + 's' if random.random()>0.5 else size_key
            d[sk] = sizes
        if use_dist and random.random() > 0.4:
            dk = dist_key if dist_key.endswith('s') else dist_key + 's' if random.random()>0.5 else dist_key
            d[dk] = distances
        if use_eps and random.random() > 0.4:
            ek = eps_key if eps_key.endswith('s') else eps_key + 's' if random.random()>0.5 else eps_key
            d[ek] = epsilons
        if not d:
            d = {pos_key: positions, size_key: sizes}
        return json.dumps(d)
    
    elif fmt == 'python_lists':
        lines = []
        if use_pos:
            lines.append(f"{pos_key} = {positions}")
        if use_size:
            lines.append(f"{size_key} = {sizes}")
        if use_dist and random.random() > 0.5:
            lines.append(f"{dist_key} = {distances}")
        if use_eps and random.random() > 0.5:
            lines.append(f"{eps_key} = {epsilons}")
        if not lines:
            lines = [f"x = {positions}", f"width = {sizes}"]
        return '\n'.join(lines)
    
    elif fmt == 'semicolon_kv':
        lines = []
        for i in range(n_entries):
            parts = []
            if use_pos:
                parts.append(f"{pos_key}={positions[i]}")
            if use_size and random.random() > 0.2:
                parts.append(f"{size_key}={sizes[i]}")
            if use_dist and random.random() > 0.5:
                parts.append(f"{dist_key}={distances[i]}")
            if use_eps and random.random() > 0.5:
                parts.append(f"{eps_key}={epsilons[i]}")
            if not parts:
                parts = [f"{pos_key}={positions[i]}", f"{size_key}={sizes[i]}"]
            lines.append('; '.join(parts))
        return '\n'.join(lines)
    
    elif fmt == 'layers_list':
        items = []
        for i in range(n_entries):
            d = {}
            attrs = [(pos_key, positions[i])]
            if use_size and random.random() > 0.3:
                attrs.append((size_key, sizes[i]))
            if use_dist and random.random() > 0.6:
                attrs.append((dist_key, distances[i]))
            random.shuffle(attrs)
            d = dict(attrs)
            items.append(d)
        s = "layers = [\n"
        for item in items:
            s += f" {item},\n"
        s = s.rstrip(',\n') + "\n]"
        if use_eps and random.random() > 0.3:
            s += f"\n{eps_key} = {epsilons}"
        return s
    
    elif fmt == 'mixed_pos_size':
        # alternate styles
        lines = []
        for i in range(n_entries):
            if random.random() > 0.5:
                lines.append(f"{'{'}'{pos_key}': {positions[i]}, '{size_key}': {sizes[i]}{'}'}")
            else:
                lines.append(f"{pos_key}: {positions[i]} | {size_key}: {sizes[i]}")
        return '; '.join(lines) if random.random()>0.5 else '\n'.join(lines)
    
    elif fmt == 'dict_with_lists':
        d = {}
        d[random.choice(['positions', 'x', 'pos'])] = positions
        d[random.choice(['thicknesses', 'widths', 'sizes'])] = sizes
        if use_dist and random.random()>0.4:
            d[random.choice(['distances', 'gaps'])] = distances
        if use_eps and random.random()>0.4:
            d[random.choice(eps_keys)] = epsilons
        return str(d)
    
    elif fmt == 'pipe_separated':
        lines = []
        for i in range(n_entries):
            parts = [f"{pos_key}: {positions[i]}"]
            if use_size:
                parts.append(f"{size_key}: {sizes[i]}")
            if use_eps and random.random()>0.5:
                parts.append(f"{eps_key}: {epsilons[i]}")
            lines.append(' | '.join(parts))
        return '\n'.join(lines)
    
    elif fmt == 'equals_lines':
        lines = []
        for i in range(n_entries):
            parts = [f"{pos_key}={positions[i]}"]
            if use_size:
                parts.append(f"{size_key}={sizes[i]}")
            if use_dist and random.random()>0.5:
                parts.append(f"{dist_key}={distances[i]}")
            lines.append('; '.join(parts))
        return '\n'.join(lines)
    
    elif fmt == 'json_string':
        items = []
        for i in range(n_entries):
            d = {pos_key: positions[i]}
            if use_size:
                d[size_key] = sizes[i]
            if use_eps and random.random()>0.4:
                d[eps_key] = epsilons[i]
            items.append(d)
        return json.dumps(items)
    
    elif fmt == 'compact_dicts':
        items = []
        for i in range(n_entries):
            d = {}
            keys = [pos_key]
            vals = [positions[i]]
            if use_size:
                keys.append(size_key)
                vals.append(sizes[i])
            if use_eps and random.random()>0.5:
                keys.append(eps_key)
                vals.append(epsilons[i])
            # compact
            items.append('{' + ', '.join(f"'{k}': {v}" for k,v in zip(keys,vals)) + '}')
        return '[' + ', '.join(items) + ']'
    
    else:  # mixed_attrs
        items = []
        for i in range(n_entries):
            d = {random.choice(pos_keys): positions[i]}
            if random.random() > 0.3:
                d[random.choice(size_keys)] = sizes[i]
            if random.random() > 0.6:
                d[random.choice(dist_keys)] = distances[i]
            if random.random() > 0.6:
                d[random.choice(eps_keys)] = epsilons[i]
            items.append(d)
        return str(items)

def generate_entry():
    data_str = generate_data_string()
    label_list = rand_int_list()
    # escape properly for the output format
    # The output is like ("data", [list])
    # data can contain quotes, so we need to handle representation
    return (data_str, label_list)

# Generate 5000
random.seed(42)
entries = []
for i in range(5000):
    entries.append(generate_entry())

# Write to file
with open('../grok5000.txt', 'w', encoding='utf-8') as f:
    for data_str, labels in entries:
        # Represent as tuple string, carefully quoting the data_str
        # Use repr for the data_str to handle escaping
        line = f"({repr(data_str)},{labels})\n"
        f.write(line)

print("Generated 5000 entries")
print("File size:", end=" ")
import os
print(os.path.getsize('../grok5000.txt'), "bytes")
