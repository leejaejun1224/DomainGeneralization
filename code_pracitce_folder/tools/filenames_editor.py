import os
import argparse

def generate_filenames(base_dir_a, base_dir_b, base_dir_c, start_idx=0, end_idx=400, ext_a='png', ext_b='png', ext_c='pfm', zero_padding=7):
    """
    Generate filenames in the format: a/0000000.png b/0000000.png c/0000000.pfm
    
    Args:
        base_dir_a (str): Base directory for the first file
        base_dir_b (str): Base directory for the second file
        base_dir_c (str): Base directory for the third file
        start_idx (int): Starting index
        end_idx (int): Ending index
        ext_a (str): File extension for the first file
        ext_b (str): File extension for the second file
        ext_c (str): File extension for the third file
        zero_padding (int): Number of zeros to pad the index
    
    Returns:
        list: List of formatted filename strings
    """
    filenames = []
    
    for idx in range(start_idx, end_idx + 1):
        # Format the index with zero padding
        idx_str = str(idx).zfill(zero_padding)
        
        # Create the filename string
        filename = f"{base_dir_a}/{idx_str}.{ext_a} {base_dir_b}/{idx_str}.{ext_b} {base_dir_c}/{idx_str}.{ext_c}"
        filenames.append(filename)
    
    return filenames

def save_filenames(filenames, output_file):
    """
    Save the generated filenames to a file
    
    Args:
        filenames (list): List of filename strings
        output_file (str): Path to the output file
    """
    with open(output_file, 'w') as f:
        for filename in filenames:
            f.write(filename + '\n')

def main():
    parser = argparse.ArgumentParser(description='Generate filenames in a specific format')
    parser.add_argument('--dir_a', type=str, default='FlyingThings3D_subset_image_clean/FlyingThings3D_subset/val/image_clean/left', help='Base directory for the first file')
    parser.add_argument('--dir_b', type=str, default='FlyingThings3D_subset_image_clean/FlyingThings3D_subset/val/image_clean/right', help='Base directory for the second file')
    parser.add_argument('--dir_c', type=str, default='FlyingThings3D_subset_disparity/FlyingThings3D_subset/val/disparity/left', help='Base directory for the third file')
    parser.add_argument('--start', type=int, default=0, help='Starting index')
    parser.add_argument('--end', type=int, default=400, help='Ending index')
    parser.add_argument('--ext_a', type=str, default='png', help='File extension for the first file')
    parser.add_argument('--ext_b', type=str, default='png', help='File extension for the second file')
    parser.add_argument('--ext_c', type=str, default='pfm', help='File extension for the third file')
    parser.add_argument('--padding', type=int, default=7, help='Number of zeros to pad the index')
    parser.add_argument('--output', type=str, default='flyingthings_val.txt', help='Output file path')
    
    args = parser.parse_args()
    
    filenames = generate_filenames(
        args.dir_a, args.dir_b, args.dir_c,
        args.start, args.end,
        args.ext_a, args.ext_b, args.ext_c,
        args.padding
    )
    
    save_filenames(filenames, args.output)
    print(f"Generated {len(filenames)} filenames and saved to {args.output}")

if __name__ == "__main__":
    main()
