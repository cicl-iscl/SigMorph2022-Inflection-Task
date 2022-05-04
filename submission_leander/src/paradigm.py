import numpy as np

from copy import deepcopy
from functools import partial
from lingpy.align.pairwise import nw_align, edit_dist


class ExactMatchScorer:
    def __getitem__(self, val):
        match = sum(c == val[1] for c in val[0])
        return match if match > 0 else -10000


class Paradigm:
    """
    Class for storing information about 1 paradigm consisting of a lemma and
    various forms with tags.
    Provides functionality for aligning forms and extracting morphological
    segments.
    """
    sep = "#"
    
    def __init__(self, lemma, forms):
        self.lemma = lemma
        self.all_forms = list(deepcopy(forms))
        self.forms = list(deepcopy(forms))
        
        self.max_epochs = 10
        
        self.scorer = ExactMatchScorer()
        self.gap = -2
        self.align = partial(nw_align, scorer=self.scorer, gap=self.gap)
        
        self.get_alignment()
        self.get_segments()
        
    @staticmethod
    def split_particles(form):
        # Split form into multiple subforms that are separated
        # by nonalphanumeric chars
        subforms = []
        
        current_subform = []
        for char in form:
            if not char.isalnum():
                current_subform = "".join(current_subform)
                    
                if current_subform:
                    subforms.append(current_subform)
                        
                current_subform = []
                
            current_subform.append(char)
            
        subforms.append("".join(current_subform))
        
        return subforms
        
    def get_alignment(self):
        # Remove particles from forms
        self.particles = set()
        no_particle_forms = set()
        
        for form in self.forms:
            subforms = self.split_particles(form)
            no_particle_forms.add(subforms[0])
            self.particles.update(set(subforms[1:]))
        
        self.forms = no_particle_forms
        
        # Sort forms in ascending order by edit distance to lemma
        # (for alignment, only the surface form of the form matters)
        self.forms.discard(self.lemma)
        self.forms = list(sorted(self.forms, key=self.dist_to_lemma))
        
        # Use lemma as anchor
        alignment = [(c,) for c in self.lemma]
        old_alignment = None
        
        done = False
        iteration = 0
        
        while not done:
            for form in self.forms:
                # If we have already aligned all forms at least once,
                # we have to remove the form before re-aligning it
                if iteration > 0:
                    alignment = [(row[0], *row[2:]) for row in alignment]
                
                # Align form to lemma and all other forms
                new_alignment, form_alignment, _ = self.align(alignment, form)
                
                # Merge alignments of forms and new form
                num_prev_forms = len(alignment[0])
                alignment = []
                
                for prev_symbs, new_symb in zip(new_alignment, form_alignment):
                    # Ignore rows that consist entirely of gaps
                    if prev_symbs == '-' and new_symb == '-':
                        continue
                    
                    # Expand gap to all aligned forms
                    if prev_symbs == '-':
                        prev_symbs = tuple(['-'] * num_prev_forms)
                    
                    # Generate new row and append to alignment
                    current_alignment_row = (*prev_symbs, new_symb)
                    alignment.append(current_alignment_row)
            
            iteration += 1
            # Stop if converged or max epochs reached
            done = ((old_alignment == alignment) or (iteration > self.max_epochs))
            old_alignment = alignment
        
        # Filter rows only consisting of gaps and transpose alignment matrix
        alignment = np.array(alignment)
        non_null_columns = (alignment != '-').sum(axis=1).nonzero()
        alignment = alignment[non_null_columns]
        
        self.alignment = alignment.T
        
        return self.alignment
    
    def get_segments(self):
        # For each column in alignment matrix, get indices of
        # forms that are aligned in that column
        aligned_idx = [(column != '-').nonzero()[0] for column in self.alignment.T]
        aligned_idx = [list(sorted(column.tolist())) for column in aligned_idx]
        
        # Get (aligned) character for each alignment column
        characters = []
        
        for column in self.alignment.T:
            # Filter gap symbols
            column = [char for char in column.tolist() if char != '-']
            # Get remaining char
            assert len(list(set(column))) == 1
            characters.append(str(column[0]))
        
        self.segments = []
        current_segment = [characters[0]]
        prev_aligned_idx = aligned_idx[0]
        
        # Always start a new segment when there is a difference
        # in the number of aligned forms for a given column
        # in the alignment matrix
        for i, current_alignment_idx in enumerate(aligned_idx[1:], 1):
            if current_alignment_idx != prev_aligned_idx:
                current_segment = "".join(current_segment)
                current_segment = {
                    'seg': current_segment,
                    'start': i - len(current_segment),
                    'end': i
                    }
                
                self.segments.append(current_segment)
                current_segment = [characters[i]]
                prev_aligned_idx = current_alignment_idx
            else:
                current_segment.append(characters[i])
        
        i = len(aligned_idx)
        current_segment = "".join(current_segment)
        current_segment = {
            'seg': current_segment,
            'start': i - len(current_segment),
            'end': i
            }
        self.segments.append(current_segment)
        
        return self.segments
    
    def form2segments(self, form):
        # Decompose form into subforms
        subforms = self.split_particles(form)
        segments = []
        
        for subform in subforms:
            if subform in self.particles:
                segments.append(subform)
            else:
                segments.extend(self.subform2segments(subform))
        
        assert "".join(segments) == form
        return segments
    
    def subform2segments(self, form):
        # Check if given form is in paradigm
        if form not in self.forms and form != self.lemma:
            raise ValueError(f"Form {form} not in this paradigm!")
        
        # Extract row from alignment matrix
        if form == self.lemma:
            form_index = 0
        else:
            form_index = self.forms.index(form) + 1
        form_alignment = self.alignment[form_index].tolist()
        
        # Collect all paradigm segments that match contiguous parts
        # of the form
        form_segments = []
        for segment in self.segments:
            form_span = form_alignment[segment['start']:segment['end']]
            form_span = "".join(form_span)
            
            if form_span == segment["seg"]:
                form_segments.append(form_span)
        
        # Check if form can be reconstructed from collected segments
        try:
            assert "".join(form_segments) == form
        except AssertionError:
            print(form_segments)
            print()
            print(form)
            print()
            print(self.segments)
            print()
            print(self.alignment)
            raise
        return form_segments

    
    def dist_to_lemma(self, form):
        return edit_dist(form, self.lemma)
    
    
if __name__ == '__main__':
    lemma = "umsteigen"
    forms = ['steigt um',
 'stiegen um',
 'steige um',
 'steigen um',
 'steigest um',
 'stiegt um',
 'stieg um',
 'stiege um',
 'steigst um',
 'steigen um',
 'umsteigen',
 'stiegen um',
 'steigen um',
 'stieg um',
 'stiegen um',
 'steige um',
 'steigt um',
 'steigen um',
 'stiegest um',
 'steig um',
 'steige um']

    
    paradigm = Paradigm(lemma, forms)
    
    print()
    print(paradigm.alignment)
    print()
    print(paradigm.segments)
    print()
    
    for form in forms:
        print(paradigm.form2segments(form))
        print()
        
