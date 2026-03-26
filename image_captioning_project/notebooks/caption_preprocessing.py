import string

# Load captions
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            image_id = tokens[0].split('.')[0]
            caption = ' '.join(tokens[1:])

            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions


def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    for key, caps in captions.items():
        for i in range(len(caps)):
            caption = caps[i]
            caption = caption.lower()
            caption = caption.translate(table)
            caption = caption.replace('  ', ' ')
            caption = 'startseq ' + caption + ' endseq'
            caps[i] = caption


# Test
filename = "../dataset/Flickr8k.token.txt"
captions = load_captions(filename)
clean_captions(captions)

print("Total images:", len(captions))
print("Sample captions:", list(captions.values())[0])
# Build vocabulary
all_words = []
for caps in captions.values():
    for cap in caps:
        all_words.extend(cap.split())

vocab = set(all_words)
print("Vocabulary Size:", len(vocab))
