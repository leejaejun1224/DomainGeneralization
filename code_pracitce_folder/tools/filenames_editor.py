input_file = "./filenames/target/driving_stereo_train_full.txt"
output_file = "./filenames/target/driving_stereo_train_full_edited.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        names = line.strip().split()
        if len(names) == 3:
            # names[0] = names[0].replace('.png', '.jpg')
            # names[1] = names[1].replace('.png', '.jpg')
            names[2] = names[2].replace('.jpg', '.png')
            outfile.write(" ".join(names) + "\n")
        else:
            print(f"Skipping line: {line.strip()}")

print("done")
