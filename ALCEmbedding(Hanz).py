#a la carte测试版本
import argparse
import os
import sys
from collections import Counter, defaultdict, OrderedDict
import numpy as np
from pathlib import Path
from unicodedata import category
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import normalize
from unicodedata import category
# 配置
GLOVEFILE = 'sgns.merge.bigram.txt'  # 替换为中文词向量文件路径

CREATE_NEW = True  # 是否为原始嵌入中已有的词创建新嵌入
FLOAT = np.float32
INT = np.uint64
SPACE = ' '

# 基本函数
def write(msg, comm=None):
    if comm is None or not comm.rank:
        sys.stdout.write(msg)
        sys.stdout.flush()
    return len(msg)

def ranksize(comm=None):
  '''returns rank and size of MPI Communicator
  Args:
    comm: MPI Communicator
  Returns:
    int, int
  '''

  if comm is None:
    return 0, 1
  return comm.rank, comm.size


def checkpoint(comm=None):
  '''waits until all processes have reached this point
  Args:
    comm: MPI Communicator
  '''

  if not comm is None:
    comm.allgather(0)

'''    
def is_punctuation(char):
  checks if unicode character is punctuation
 

  return category(char)[0] in CATEGORIES
'''

def subtokenize(token, vocab):
    return [token] if token in vocab else [False]


class ALaCarteReader:
  '''reads documents and updates context vectors
  '''

  def __init__(self, w2v, targets, wnd=10, checkpoint=None, interval=[0, float('inf')], comm=None):
    '''initializes context vector dict as self.c2v and counts as self.target_counts
    Args:
      w2v: {word: vector} dict of source word embeddings
      targets: iterable of targets to find context embeddings for
      wnd: context window size (uses this number of words on each side)
      checkpoint: path to HDF5 checkpoint file (both for recovery and dumping)
      interval: corpus start and stop positions
      comm: MPI Communicator
    '''

    self.w2v = w2v
    self.combined_vocab = self.w2v
    gramlens = {len(target.split()) for target in targets if target}
    self.max_n = max(gramlens)
    if self.max_n > 1:
      self.targets = [tuple(target.split()) for target in targets]
      self.target_vocab = set(self.targets)
      self.combined_vocab = {word for target in targets for word in target.split()}.union(self.combined_vocab)
    else:
      self.targets = targets
      self.target_vocab = set(targets)
      self.combined_vocab = self.target_vocab.union(self.combined_vocab)
    self.target_counts = Counter()

    dimension = next(iter(self.w2v.values())).shape[0]
    self.dimension = dimension
    self.zero_vector = np.zeros(self.dimension, dtype=FLOAT)
    self.c2v = defaultdict(lambda: np.zeros(dimension, dtype=FLOAT))


    self.wnd = wnd
    self.learn = len(self.combined_vocab) == len(self.target_vocab) and self.max_n == 1
    self.processed_tokens = set()  # 新增：跟踪处理过的词汇

    self.datafile = checkpoint
    self.comm = comm
    self.rank, self.size = ranksize(comm)
    position = interval[0]

    if self.rank:
      self.vector_array = FLOAT(0.0)
      self.count_array = INT(0)

    elif checkpoint is None or not os.path.isfile(checkpoint):
      self.vector_array = np.zeros((len(self.targets), dimension), dtype=FLOAT)
      self.count_array = np.zeros(len(self.targets), dtype=INT)

    else:

      import h5py

      f = h5py.File(checkpoint, 'r')
      position = f.attrs['position']
      assert interval[0] <= position < interval[1], "checkpoint position must be inside corpus interval"
      self.vector_array = np.array(f['vectors'])
      self.count_array = np.array(f['counts'])

    self.position = comm.bcast(position, root=0) if self.size > 1 else position
    self.stop = interval[1]

  def reduce(self):
    '''reduces data to arrays at the root process
    '''

    comm, rank, size = self.comm, self.rank, self.size
    targets = self.targets

    c2v = self.c2v
    dimension = self.dimension
    vector_array = np.vstack([c2v.pop(target, np.zeros(self.dimension, dtype=FLOAT)) for target in self.targets])

    target_counts = self.target_counts
    count_array = np.array([target_counts.pop(target, 0) for target in targets], dtype=INT)

    if rank:
      comm.Reduce(vector_array, None, root=0)
      comm.Reduce(count_array, None, root=0)
    elif size > 1:
      comm.Reduce(self.vector_array + vector_array, self.vector_array, root=0)
      comm.Reduce(self.count_array + count_array, self.count_array, root=0)
    else:
      self.vector_array += vector_array
      self.count_array += count_array

  def checkpoint(self, position):
    '''dumps data to HDF5 checkpoint
    Args:
      position: reader position
    Returns:
      None
    '''

    datafile = self.datafile
    assert not datafile is None, "no checkpoint file specified"
    self.reduce()

    if not self.rank:

      import h5py

      f = h5py.File(datafile + '~tmp', 'w')
      f.attrs['position'] = position
      f.create_dataset('vectors', data=self.vector_array, dtype=FLOAT)
      f.create_dataset('counts', data=self.count_array, dtype=INT)
      f.close()
      if os.path.isfile(datafile):
        os.remove(datafile)
      os.rename(datafile + '~tmp', datafile)
    self.position = position

  def target_coverage(self):
    '''returns fraction of targets covered (as a string)
    Args:
      None
    Returns:
      str (empty on non-root processes)
    '''

    if self.rank:
      return ''
    return str(sum(self.count_array > 0)) + '/' + str(len(self.targets))

  def read_ngrams(self, tokens):
    '''reads tokens and updates context vectors
    Args:
      tokens: list of strings
    Returns:
      None
    '''

    import nltk

    # gets location of target n-grams in document
    target_vocab = self.target_vocab
    max_n = self.max_n
    ngrams = dict()
    for n in range(1, max_n + 1):
      ngrams[n] = list(filter(lambda entry: entry[1] in target_vocab, enumerate(nltk.ngrams(tokens, n))))

    for n in range(1, max_n + 1):
      if ngrams[n]:

        # gets word embedding for each token
        w2v = self.w2v
        zero_vector = self.zero_vector
        wnd = self.wnd
        start = max(0, ngrams[n][0][0] - wnd)
        vectors = [None] * start + [w2v.get(token, zero_vector) if token else zero_vector for token in
                                    tokens[start:ngrams[n][-1][0] + n + wnd]]
        c2v = self.c2v
        target_counts = self.target_counts

        # computes context vector around each target n-gram
        for i, ngram in ngrams[n]:
          c2v[ngram] += sum(vectors[max(0, i - wnd):i], zero_vector) + sum(vectors[i + n:i + n + wnd],
                                                                           zero_vector)
          target_counts[ngram] += 1

  def read_document(self, document):
    print("Processing document:", document)
    tokens = document.split()
    for i, token in enumerate(tokens):
        # Check if token is in combined vocab or if we are learning new embeddings
        if token in self.combined_vocab or self.learn:
            start = max(0, i - self.wnd)
            end = min(len(tokens), i + self.wnd + 1)
            context_tokens = tokens[start:i] + tokens[i+1:end]
            self.processed_tokens.add(token) 
            context_vectors = [self.w2v.get(t, self.zero_vector) for t in context_tokens]

            # Adding debug information
            if not context_vectors:
                print(f"No context vectors for token: {token}")
            else:
                for vec in context_vectors:
                    if vec.shape != (300,):
                        print(f"Incorrect shape for vector: {vec.shape}")

            context_vector = np.mean(context_vectors, axis=0) if context_vectors else self.zero_vector

            # Ensure correct shape for context_vector
            if context_vector.shape != (300,):
                print(f"Incorrect shape for context vector of '{token}': {context_vector.shape}")
                context_vector = np.reshape(context_vector, (300,))

            # Update context vectors and target counts
            self.c2v[token] += context_vector
            self.target_counts[token] += 1
        else:
            print(f"Token '{token}' not found in combined vocabulary. ")
      



def make_printable(string):
    if not isinstance(string, str):
        print("Warning: Non-string object passed to make_printable.")
        return string  # 或者返回空字符串，根据您的需求决定
    return ''.join(filter(str.isprintable, string))


def process_documents(func):
  '''wraps document generator function to handle English-checking and lower-casing and to return data arrays
  '''

  def wrapper(string, reader, verbose=False, comm=None, english=False, lower=False):

    generator = (make_printable(document) for document in func(string, reader, verbose=verbose, comm=comm))
    

    for i, document in enumerate(generator):
      reader.read_document(document)

    reader.reduce()
    write('\rFinished Processing Corpus; Targets Covered: ' + reader.target_coverage() + ' \n', comm)
    return reader.vector_array, reader.count_array

  return wrapper




#@process_documents
def corpus_documents(corpusfile, reader, verbose=False, comm=None):
    position = reader.position
    rank, size = ranksize(comm)
    with open(corpusfile, 'r') as f:
        f.seek(position)
        for i, line in enumerate(f):
            stripped_line = line.strip()
            if not isinstance(stripped_line, str):
                print(f"Warning: Line {i} is not a string, but {type(stripped_line)}")
            print(f"Document type: {type(line.strip())}")
            yield line.strip()
            
            if i < position:
                continue
            if i >= reader.stop:
                break
            if i % 1000000 == 0 and verbose and not rank:
                reader.reduce()
                write('\rProcessed ' + str(i) + ' Lines; Target Coverage: ' + reader.target_coverage(), comm)
                if not reader.datafile is None:
                    reader.checkpoint(i)
            if i % size == rank:
                yield line.strip()  # 确保这里返回的是字符串


def load_vectors(vectorfile):
    words = set()
    with open(vectorfile, 'r') as f:
        for line in f:
            parts = line.strip().split(SPACE)
            word = parts[0]
            vector_parts = parts[1:]

            # 检查是否有足够的部分构成一个向量
            if len(vector_parts) != 300:
                print(f"Warning: Skipping word '{word}' due to incorrect number of vector parts: {len(vector_parts)}")
                continue

            try:
                vector = np.array([float(p) for p in vector_parts], dtype=FLOAT)
            except ValueError as e:
                print(f"Error converting vector parts to floats for word '{word}': {e}")
                continue

            if word not in words:
                words.add(word)
                yield word, vector


def dump_vectors(generator, vectorfile):
  '''dumps embeddings to .txt
  Args:
    generator: (gram, vector) generator; vector can also be a scalar
    vectorfile: .txt file
  Returns:
    None
  '''

  with open(vectorfile, 'w') as f:
    for gram, vector in generator:
      numstr = ' '.join(map(str, vector.tolist())) if vector.shape else str(vector)
      f.write(gram + ' ' + numstr + '\n')


def dump_targets(generator, targetfile):
  with open(targetfile, 'w') as f:
    f.writelines([f'{target}\n' for target in generator])

def extract_unique_words(corpus_file):
    """从给定的语料库文件中提取所有唯一的词汇"""
    unique_words = set()
    with open(corpus_file, 'r') as f:
        for line in f:
            words = line.strip().split()
            unique_words.update(words)
    return list(unique_words)

def parse():
    '''解析命令行参数'''

    parser = argparse.ArgumentParser(prog='python alacarte.py')

    parser.add_argument('dumproot', help='中间和输出文件的根文件名', type=str)
    parser.add_argument('-m', '--matrix', help='a la carte变换矩阵的二进制文件', type=str)
    parser.add_argument('-v', '--verbose', action='store_true', help='显示进度')
    parser.add_argument('-r', '--restart', help='用于重启的HDF5检查点文件', type=str)
    parser.add_argument('-i', '--interval', nargs=2, default=['0', 'inf'], help='语料库位置区间')

    parser.add_argument('-s', '--source', default=GLOVEFILE, help='源词嵌入文件', type=str)
    parser.add_argument('-c', '--corpus', nargs='*', help='文本语料库文件列表')
    parser.add_argument('-t', '--targets', help='目标词文件', type=str)

    parser.add_argument('-w', '--window', default=10, help='上下文窗口大小', type=int)
    parser.add_argument('--create-new', action='store_true', help='为原始嵌入中已有的词创建新嵌入')

    return parser.parse_args()


    

def run_alacarte(source_embeddings_file, targets_file, corpus_file, matrix_file, dumproot,outputname, window_size=10, create_new=True, restart_file=None, interval=[0, float('inf')]):
    # 加载词向量
    w2v = OrderedDict(load_vectors(source_embeddings_file))

    # 加载目标词
    if targets_file:
        with open(targets_file, 'r') as f:
            targets = [target.strip() for target in f if not target.strip() in w2v or create_new]
    else:
        targets = extract_unique_words(corpus_file)

    assert len(targets), "No targets found"

    # 加载变换矩阵
    if os.path.isfile(matrix_file):
        M = np.fromfile(matrix_file, dtype=FLOAT).reshape(-1, len(w2v[list(w2v.keys())[0]]))
    else:
        M = None

    # 初始化 ALaCarteReader
    alc = ALaCarteReader(w2v, targets, wnd=window_size, checkpoint=restart_file, interval=interval)
    # 从语料库构建上下文向量
    
    for document in corpus_documents(corpus_file, alc):
        assert isinstance(document, str), f"Expected string, got {type(document)}"
        alc.read_document(document)
    # 在处理完所有文档后，获取上下文向量和目标计数
    context_vectors = np.array([alc.c2v[target] for target in targets])
    target_counts = np.array([alc.target_counts[target] for target in targets])
    # 检查目标单词的覆盖情况
    nz = target_counts > 0
       
    if M is None:
        from sklearn.linear_model import LinearRegression as LR
        from sklearn.preprocessing import normalize
        # 初始化 X 和 Y 的列表
        X_list, Y_list = [], []

        for target in targets:
            if target in w2v and alc.target_counts[target] > 0:
                X_list.append(alc.c2v[target])
                Y_list.append(w2v[target])

        # 转换列表为 NumPy 数组
        X = np.array(X_list)
        Y = np.array(Y_list)

        # 检查 X 和 Y 是否具有相同的样本数
        assert X.shape[0] == Y.shape[0], f"Inconsistent number of samples in X and Y: {X.shape[0]} vs {Y.shape[0]}"

        M = LR(fit_intercept=False).fit(X, Y).coef_.astype(FLOAT)

        # 保存变换矩阵和上下文向量
        M.tofile(matrix_file)
        context_vectors.tofile(dumproot + f'{outputname}_source_context_vectors.bin')
        dump_vectors(zip(targets, target_counts), dumproot + f'{outputname}_source_vocab_counts.txt')
    induced_vectors = [(target, alc.c2v[target].dot(M.T)) for target in targets if alc.target_counts[target] > 0]
    dump_vectors(induced_vectors, dumproot + f'{outputname}_alacarte.txt')
    
    # 导出未找到的目标词
    targets_not_found = [target for target in targets if alc.target_counts[target] == 0]
    dump_targets(targets_not_found, dumproot + f'{outputname}_not_found.txt')


from scipy.spatial.distance import cosine

def load_word_vectors(filename):
    """
    从文本文件中加载词向量。
    """
    word_vectors = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            word_vectors[word] = vector
    return word_vectors



def find_nearest_neighbors(word_vectors, target_word, top_n=5):
    """
    找到与目标词最近的n个邻居，修正版。
    """
    if target_word not in word_vectors:
        return "Target word not found in word vectors."
    
    target_vector = word_vectors[target_word]
    distances = []
    
    for word, vector in word_vectors.items():
        if word == target_word or np.all(vector == 0):
            continue
        # 计算余弦距离，注意这里直接使用cosine函数，它返回的是距离而不是相似度
        distance = cosine(target_vector, vector)
        if not np.isnan(distance):  # 过滤掉计算结果为NaN的情况
            distances.append((word, distance))
    
    # 按距离排序并获取最近的n个词，这里不再转换为距离，因为cosine函数直接返回距离
    nearest_neighbors = sorted(distances, key=lambda x: x[1])[:top_n]
    
    return nearest_neighbors
