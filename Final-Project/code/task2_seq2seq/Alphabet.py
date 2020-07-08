import torch
class Alphabet(object):
    def __init__(self, filelist, store = False, load = False):
        self.alphabet = {}
        self.index = []

        if load:
            with open('Alphabet.txt','r') as f:
                textline = f.readlines()
                for texts in textline:
                    char, i = texts.rstrip('\n').split()
                    self.alphabet[char] = int(i)
                    self.index.append(char)
        else:
            for files in filelist:
                with open(files,'r') as f:
                    textline = f.readlines()
                    if textline == []:
                        textline.append('')
                    assert len(textline) ==1
                    textline = list(textline[0])
                    self.index+=textline
            self.index = list(set(self.index))

            for i, char in enumerate(self.index):
                self.alphabet[char] = i + 1

        
        if store:
            with open('Alphabet.txt','w') as f:
                for key, value in self.alphabet.items():
                    f.write(key + ' ' + str(value) + '\n')
                pass
    def blank(self):
        idx = len(self.alphabet)
        self.index.append('<BLANK>')
        self.alphabet['<BLANK>'] = idx
        return idx

    def length(self):
        return len(self.index)
    
    def encode(self, text):
        length = []
        result = []
        for item in text:
            item_len = 0
            for char in item:
                if char != '<BLANK>':
                    index = self.alphabet[char]
                    result.append(index)
                    item_len += 1
            length.append(item_len)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))
    
    def decode(self, t, length, raw = False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
