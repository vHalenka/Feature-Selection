import os

# Directories for animations and noise distributions
animation_dir = "animations"
noise_distribution_dir = "noise_distributions"
output_html = "index.html"

# Extract groups for organizing animations
def extract_group(filename):
    parts = filename.rsplit("_", 1)  # Split by the last underscore
    return parts[0] if len(parts) > 1 else "other"

# Generate HTML
def generate_html(animation_dir, noise_distribution_dir, output_html):
    groups = {}
    
    # Group animations
    for filename in os.listdir(animation_dir):
        if filename.endswith(".mp4"):
            group = extract_group(filename)
            if group not in groups:
                groups[group] = []
            groups[group].append(filename)

    # Open HTML file for writing
    with open(output_html, "w") as f:
        f.write("<html><head><title>Experiment Results</title></head><body>\n")
        f.write("<h1>Experiment Results</h1>\n")
        
        # Add noise distribution section
        f.write("<h2>Noise Distributions</h2>\n")
        for plot_file in sorted(os.listdir(noise_distribution_dir)):
            if plot_file.endswith(".png"):
                f.write(f'<div style="margin-bottom: 20px;">\n')
                f.write(f'<h3>{plot_file.split(".")[0]}</h3>\n')
                f.write(f'<img src="{noise_distribution_dir}/{plot_file}" alt="{plot_file}" style="width: 600px;">\n')
                f.write(f'</div>\n')

        # Add animations section
        f.write("<h2>Animations</h2>\n")
        for group, files in sorted(groups.items()):
            f.write(f"<h3>{group}</h3>\n")
            f.write('<div style="display: flex; flex-wrap: wrap;">\n')
            for filename in sorted(files):
                f.write(f'<div style="margin: 10px;">\n')
                f.write(f'<video width="300" controls>\n')
                f.write(f'  <source src="{animation_dir}/{filename}" type="video/mp4">\n')
                f.write("  Your browser does not support the video tag.\n")
                f.write("</video>\n")
                f.write(f'<p>{filename}</p>\n')
                f.write("</div>\n")
            f.write("</div>\n")
        
        f.write("</body></html>\n")
    print(f"HTML file generated: {output_html}")

# Generate the HTML file
generate_html(animation_dir, noise_distribution_dir, output_html)
