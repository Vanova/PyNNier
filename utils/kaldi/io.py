import numpy as np

START_ARK_MARK = '['
END_ARK_MARK = ']'


class ArkReader(object):
    def __init__(self, data_file):
        self.data_file = data_file

    @staticmethod
    def grep(data_file, pattern='['):
        with open(data_file, 'r') as file:
                match = [line for line in file if pattern in line]
        return match

    def next_ark(self):
        """
        Read file with arks,
        return every ark file one-by-one
        """
        cnt = 0
        with open(self.data_file, 'r') as file:
            utt = ''
            feats = []
            for line in file:
                # start of the features
                if '[' in line:
                    utt = line.strip().split()
                    utt.remove(START_ARK_MARK)
                    utt = utt[0]
                    cnt += 1
                # end of the features
                elif ']' in line:
                    tmp = line.strip().split()
                    tmp.remove(END_ARK_MARK)
                    feats.append(tmp)
                    # return current ark file
                    yield utt, str2float_array(feats)
                    utt = ''
                    feats = []
                else:
                    # read features
                    feats.append(line.strip().split())
        # print("Read %d files from: %s" % (cnt, self.data_file))


def read_ark_file(file_name):
    """
    Read all ark files in memory inside storage 'file_name',
    lazy approach
    """
    list_utt_feat = []
    cnt = 0
    with open(file_name, 'r') as file:
        utt = ''
        feats = []
        for line in file:
            # start of the features
            if '[' in line:
                utt = line.strip().split()
                utt.remove(START_ARK_MARK)
                utt = utt[0]
                cnt += 1
            # end of the features
            elif ']' in line:
                tmp = line.strip().split()
                tmp.remove(END_ARK_MARK)
                feats.append(tmp)
                # save result
                list_utt_feat.append((utt, str2float_array(feats)))
                utt = ''
                feats = []
            else:
                # read features
                feats.append(line.strip().split())
    print("Read %d files..." % cnt)
    return list_utt_feat


def write_ark_file(list_utt_feat, file_name):
    cnt = 0
    with open(file_name, 'w') as file:
        for utt, feats in list_utt_feat:
            head = "%s  [" % utt
            txt_feat = "\n  ".join(" ".join(str(i) for i in x) for x in feats)
            file.write("%s \n  %s ]\n" % (head, txt_feat))
            cnt += 1
    print("Wrote %d files..." % cnt)


def read_mlf(fname):
    with open(fname, 'r') as file:
        res = {}
        ut_id = None
        alignment = []
        for line in file:
            if '.lab' in line:  # utterance file name
                ut_id = line.strip()
            elif '.' == line.strip():  # end of utterance
                res[ut_id] = alignment
            else:
                sep = line.strip()
                sep = sep.split(' ')
                alignment.append(sep)
        return res


def str2float_array(arr):
    return np.array(arr).astype(np.float)


if __name__ == "__main__":
    file_name = 'test.ark'
    ark = read_ark_file(file_name)
    file_name = 'test_out.ark'
    write_ark_file(ark, file_name)
    file_name = 'phone.mlf'
    mlf = read_mlf(file_name)
