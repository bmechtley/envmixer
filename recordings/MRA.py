"""
Recording/MRA.py
envmixer

2012 Brandon Mechtley
Arizona State University
"""

import pywt
import numpy as np
from .Recording import Recording

class MRANode:
    """
    MRA node object used by MRATree with links to ancestors and predecessors.
    
    Each MRA node represents a single coefficient of a multiresolution discrete
    wavelet transform.
    """
    
    def __init__(self, value, parent, predecessor, side, level):
        self.value = value
        self.children = [None, None]
        self.parent = parent
        self.predecessor = predecessor
        self.level = level
        if self.parent is not None: self.parent.children[side] = self
    
    def __str__(self):
        return 'MRANode: %d, level %d' % (self.value, self.level)
    
    def childIndex(self):
        """Return the index of this node in its parent's list of children."""
        
        return self.parent.children.index(self)
    
    def ancestors(self):
        """Return a list of ancestors, going all the way to the root."""
        
        ancestor_list = []
        n = self
        
        while n.parent:
            ancestor_list.append(n.parent)
            n = n.parent
        
        return ancestor_list
    
    def predecessors(self, k = 2):
        """Return a list of predecessors of length k."""
        
        predecessor_list = []
        n = self
        
        for i in range(0, k):
            predecessor_list.append(n.predecessor)
            n = n.predecessor
        
        return predecessor_list
    
    def adopt(self, n1, n2):
        """Give up children and adopt two new ones (n1 and n2)."""
        
        self.children = [n1, n2]
        n1.parent = self
        n2.parent = self

class MRA(Recording):
    """
    MRA tree creation/manipulation functions for a Recording.
    """
    
    def __init__(self, filename):
        super(Recording, self).__init__(filename)

    def calculate_mra(self, wavelet='db10', mode='per'):
        """
        Creates an MRA wavelet tree on the recording.
        
        Args:
            wavelet (str): wavelet to use. Any string supported by PyWavelets will work.
            mode (str): method for handling overrun. Default "per," start over at the beginning of the waveform
                (periodic).
        """
        
        self.wavelet, self.mode = wavelet, mode
        self.dwt = pywt.wavedec(self.wav, wavelet, mode=mode, level=int(np.log2(len(self.wav))) + 1)
        
        self.root = None
        self.nodes = []
        self.wavelet = wavelet
        self.mode = mode
        parents = [None]

        for i in range(len(self.dwt)):
            nodes = []

            for j in range(len(self.dwt[i])):
                if j > 0:
                    nodes.append(MRANode(self.dwt[i][j], parents[j / 2], nodes[j - 1], j % 2, i))
                else:
                    nodes.append(MRANode(self.dwt[i][j], parents[j / 2], None, j % 2, i))

            nodes[0].predecessor = nodes[-1]
            parents = nodes
            self.nodes.extend(nodes)
            if i is 0: self.root = nodes[0]

    def reconstruct_wav(self, node=None, level=0, index=0):
        """
        Bake an MRA into a one-dimensional waveform.
        """
        
        if node is None: node = self.root
        self.dwt[level][index] = node.value

        map(
            lambda c:
                self.reconstruct(c, level + 1, index * 2 + c.parent.children.index(c))
                if c else None,
            node.children
        )

        if not level:
            return pywt.pywt.waverec(self.dwt, self.wavelet, mode=self.mode)

    def ascore(self, c, nancestors, threshold):
        """
        Return wavelet tree learning ancestor similarity score.
        
        c -- candidate node in question.
        nancestors -- list of the original node's ancestors (MRANode instances).
        threshold -- value under which two paths are considered similar. 
        """
        
        cancestors = array([p.value for p in c.ancestors()])
        ascores = abs(nancestors - cancestors)
        ascores /= linspace(1, len(ascores), len(ascores))
        return sum(cumsum(ascores) < threshold)
        
    def pscore(self, c, npredecessors, nk, threshold):
        """Return wavelet tree learning predecessor similarity score.
        
        c -- candidate node in question.
        npredecessors -- list of the original node's predecessors.
        nk -- number of predecessors to consider.
        threshold -- value under which two paths are considered similar.
        """
        
        cpredecessors = array([p.value for p in c.predecessors(nk)])
        pscores =  abs(npredecessors - cpredecessors)
        return sum(cumsum(pscores) < threshold)
    
    def tap(self, p = 0.8, k = 0.01, maxlevel = -1):
        """
        Rearrange tree using wavelet tree learning as per TAPESTREA.
        
        p -- what percentage of nodes to consider for the threshold. 
            Larger values result in more variation.
        k -- number of predecessors to consider in comparing node contexts.
            Smaller values will tend to break up longer events.
        maxlevel -- maximum level at which to rearrange nodes. Oftentimes
            10 is sufficient. Use -1 (default) to rearrange all levels.
            Note that children of rearranged nodes will be moved around
            regardless.
        """
        
        threshold = scoreatpercentile([abs(n.value) for n in self.nodes], p * 100) * 2.0
        parents = array([])
        children = array([self.root])
        
        while len(children) > 0:
            nodes = children.copy()
            new_nodes = nodes.copy()
            level = nodes[0].level
            
            if level > 2 and (level < maxlevel or maxlevel < 0):                
                nk = int(k * 2 ** level)
                
                # Choose new node ordering.
                for i in range(0, len(nodes)):
                    nancestors = array([p.value for p in nodes[i].ancestors()])
                    npredecessors = array([p.value for p in nodes[i].predecessors(nk)])
                    
                    ascores = array([self.ascore(c, nancestors, threshold) for c in nodes])
                    pscores = array([self.pscore(c, npredecessors, nk, threshold) for c in nodes])
                    
                    candidates = nodes[(ascores == max(ascores)) & (pscores == max(pscores))]
                    selection = choice(candidates)
                    new_nodes[i] = selection
                
                # Fix predecessor relationships.
                for i in range(0, len(new_nodes)):
                    new_nodes[i].predecessor = new_nodes[i - 1]
                
                # Fix parent/child relationships.
                for i in range(0, len(parents)):
                    parents[i].adopt(new_nodes[i * 2], new_nodes[i * 2 + 1])
            
            parents = nodes
            children = array([c for c in array([n.children for n in nodes]).flat if c != None])
            