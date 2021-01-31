import pandas as pd
import spacy
import re
import string
from itertools import zip_longest

pd.options.mode.chained_assignment = None
### Tree Method


class TokenClassifier:

    def __init__(self, path_to_file,
                 col: 'Column to parse'):
        self.nlp = spacy.load("en_core_web_sm")
        self.path = path_to_file
        self.df = self.clean()
        self.serie = self.df[col]
        self._sentize()
        self.df['_parsed_sents'] = self.df['_sents'].apply(self.explore)
        self._gencol()


    def clean(self):
        df = pd.read_excel(self.path, encoding='UTF-8')




        pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})\s-\s([a-zA-Z]+\s?[a-zA-Z]+\s?[a-zA-Z]{,5})\s(?:- Vendor )?\((\D*)\)\r?\n((?:(?!\d{4}-\d{2}-\d{2}).)*)",
            re.DOTALL)
        task_pattern = re.compile(r"(TASK\d{6,9}):\s?")
        alphabet = list(string.ascii_lowercase)

        def task_col(text, **kwargs):
            converted = str(text)
            for val in kwargs.values():
                if val == 1:
                    if re.search(task_pattern, converted):
                        return re.search(task_pattern, converted).group(1)
                elif val == 2:
                    if re.search(task_pattern, converted):
                        return re.sub(task_pattern, '', text)
                    else:
                        return text

        data = []
        for i, id, comms, *_ in df.itertuples():
            for chunk in re.findall(pattern, comms):
                data.append((id,) + chunk)

        wlf = pd.DataFrame(data, columns=['ID', 'Date', 'Time', 'Author', 'Type', 'Notes'])

        wlf['task_no'] = wlf['Notes'].apply(task_col, mode=1)
        wlf['Notes'] = wlf['Notes'].apply(task_col, mode=2)

        return wlf[wlf['Notes'].notnull()].reset_index(drop=True)

    def _separate(self):
        for row in self.serie:
            self.part = row
            yield self.part

    def _docize(self):
        for s_part in self._separate():
            self.doc = self.nlp(s_part)
            yield self.doc

    def _sentize(self):
        sentences = []
        for block in self._docize():
            blocks = []
            for sents in block.sents:
                blocks.append(sents)
            sentences.append(blocks)
        self.df['_sents'] = sentences
        self.df['_sents_no'] = self.df['_sents'].map(lambda x: len(x))


    def tokenize(self, sent):
        d = {}
        dep_d = {}
        pos_d = {}
        token = sent.root

        def tok(token):
            if token.dep_ != 'punct':
                if len([child for child in token.children]) == 0:
                    ancestor_line = [token] + [t for t in token.ancestors]
                    d.setdefault(ancestor_line[-1].text, {})  # Text
                    dep_d.setdefault(ancestor_line[-1].dep_, {})  # Dep
                    pos_d.setdefault(ancestor_line[-1].pos_, {})  # POS

                    if len(ancestor_line) >= 2:
                        d[ancestor_line[-1].text].setdefault(ancestor_line[-2].text, [])  # Text
                        [d[ancestor_line[-1].text][ancestor_line[-2].text].append(k.text) for k in
                         ancestor_line[:-2]]  # Text
                        dep_d[ancestor_line[-1].dep_].setdefault(ancestor_line[-2].dep_, [])  # Dep
                        [dep_d[ancestor_line[-1].dep_][ancestor_line[-2].dep_].append(k.dep_) for k in
                         ancestor_line[:-2]]  # Dep
                        pos_d[ancestor_line[-1].pos_].setdefault(ancestor_line[-2].pos_, [])  # POS
                        [pos_d[ancestor_line[-1].pos_][ancestor_line[-2].pos_].append(k.pos_) for k in
                         ancestor_line[:-2]]  # POS


                    else:
                        [d[ancestor_line[-1].text].append(k.text) for k in ancestor_line[:-1]]  # Text
                        [dep_d[ancestor_line[-1].dep_].append(k.dep_) for k in ancestor_line[:-1]]  # Dep
                        [pos_d[ancestor_line[-1].pos_].append(k.pos_) for k in ancestor_line[:-1]]  # POS



                else:

                    for child in token.children:
                        tok(child)

        tok(token)
        return d, dep_d, pos_d

    def explore(self, val, *args):



        def depth_search(dic, depth):

            if depth == 0:
                return [key for key in dic.keys()]

            elif depth == 1:
                return [key for value in dic.values() for key in value.keys()]

            elif depth == 2:
                return [val for value in dic.values() for val in value.values()]

        schema = {}

        for r, sent in enumerate(val):

            schema[r] = {}

            data, dep_data, pos_data = self.tokenize(sent)


            # SUBJECT



            subject_dep = []
            subject_text = []
            for p_i, part in enumerate(depth_search(dep_data, 1)):
                if (part == 'nsubj') | (part == 'nsubjpass'):
                    # Get LEVEL 1 text and dep

                    try:
                        if sent.root.pos_ == 'NOUN':
                            subject_dep.append(part)
                            subject_text.append(depth_search(data, 0)[p_i])
                        subject_dep.append(part)
                        subject_text.append(depth_search(data, 1)[p_i])

                    except IndexError:
                        pass
                    # Get LEVEL 2 text and dep

                    dep_level_2 = dep_data['ROOT'].get(part)
                    text_level_2 = depth_search(data, 2)[depth_search(dep_data, 2).index(dep_level_2)]

                    # Create unordered list of subject elements

                    subject_dep.append(dep_level_2)  # Dep
                    [subject_text.append(text_item) for text_item in text_level_2]

                    # Returning the ordered subject string
                    schema[r]['subject'] = ' '.join(
                        [s_tok.text for s_tok in sent if s_tok.text in subject_text])




            # PREDICATE

            # Decide between VERB or NOUN ROOT


            if (sent.root.pos_ == 'VERB') | (sent.root.pos_ == 'AUX'):
                # Get LEVEL 0 text and dep
                predicate_dep = []
                predicate_text = []

                predicate_dep.append(sent.root.dep_)
                predicate_text.append(sent.root.text)

                # Get LEVEL 1 root acompanying text and dep

                for p_i, part in enumerate(depth_search(dep_data, 1)):
                    if (part == 'advmod') | (part == 'aux') | (part == 'neg') | (part == 'auxpass'):
                        # Get LEVEL 1 text and dep

                        predicate_dep.append(part)
                        predicate_text.append(depth_search(data, 1)[p_i])

                    schema[r]['predicate'] = ' '.join(
                        [p_tok.text for p_tok in sent if p_tok.text in predicate_text])
                    schema[r]['predicate_lemma'] = ' '.join(
                        [p_tok.lemma_ for p_tok in sent if p_tok.text in predicate_text])



            # OBJECT
            object_dep = []
            object_text = []
            for p_i, part in enumerate(depth_search(dep_data, 1)):

                if (part == 'prep') | (part == 'pobj') | \
                   (part == 'dobj') | (part == 'advcl') | \
                   (part == 'xcomp') | (part == 'acomp') | \
                   (part == 'ccomp'):
                    # Get LEVEL 1 text and dep

                    try:
                        object_dep.append(part)
                        object_text.append(depth_search(data, 1)[p_i])

                        # Get LEVEL 2 text and dep

                        dep_level_2 = dep_data['ROOT'].get(part)
                        text_level_2 = depth_search(data, 2)[depth_search(dep_data, 2).index(dep_level_2)]

                        # Create unordered list of subject elements

                        object_dep.append(dep_level_2)  # Dep
                        [object_text.append(text_item) for text_item in text_level_2]

                        # Returning the ordered subject string
                        schema[r]['object'] = ' '.join(
                            [o_tok.text for o_tok in sent if o_tok.text in object_text])

                    except IndexError:
                        pass

        subject_dict = {}
        predicate_dict = {}
        obj_dict = {}
        for arg in args:
            if arg == 'subject':
                for k in schema.keys():
                    subject_dict[k] = schema[k].setdefault('subject','')
                return subject_dict
            elif arg == 'predicate':
                for k in schema.keys():
                    predicate_dict[k] = schema[k].setdefault('predicate', '')
                return predicate_dict
            elif arg == 'obj':
                for k in schema.keys():
                    obj_dict[k] = schema[k].setdefault('object', '')
                return obj_dict

                pass
            else:
                pass

        return schema


    def _gencol(self):

        def gen(pt):
            count = 0
            for k,v in pt.items():


                if v != {}:
                    count += 1


                    self.df.loc[t,f'{str(count).zfill(3)}_subject'] = pt[k].setdefault('subject','')
                    self.df.loc[t,f'{str(count).zfill(3)}_predicate'] = pt[k].setdefault('predicate','')
                    self.df.loc[t,f'{str(count).zfill(3)}_predicate_lemma'] = pt[k].setdefault('predicate_lemma','')
                    self.df.loc[t,f'{str(count).zfill(3)}_object'] = pt[k].setdefault('object','')


                else:
                    pass
            return count

        for t, item in enumerate(self.df['_parsed_sents']):
            self.df.loc[t,'_valid_sent_no'] = gen(item)





    def return_df(self):
        return self.df



class Explore:
    def __init__(self,db):
        self.db = db




    def fetch(self, get=None, **kwargs):

        get_it = get
        #Getting keywords
        subject = pred_placeholder = predicate = object = None
        for key,value in kwargs.items():
            arg_list = list(kwargs.keys())
            if value:
                if key == 'subject':
                    subject = value
                elif key == 'predicate':
                    predicate = value
                elif key == 'object':
                    object = value

            else:
                value = None
        value_list = [subject, pred_placeholder, predicate, object]


        gen = (x for x in self.db.columns if x.startswith('0'))
        downsize_df = self.db[['ID'] + list(gen)]
        extract_df = self.db.iloc[:,:6]
        f_val = sum(1 for i in value_list if i != None)


        def return_search_df():

            def grouper(iterable, n, fillvalue=None):
                args = [iter(iterable)] * n
                return zip_longest(*args, fillvalue=fillvalue)
            ind = []
            for t,id,*cols in downsize_df.itertuples():
                ind.append(False)
                for k in grouper([*cols],4):

                    check_sum = sum(1 for i in range(len(value_list)) if (str(value_list[i]).lower() in str(k[i]).lower()))
                    if check_sum == f_val:
                        ind.pop()
                        ind.append(True)
                        break

            return ind

        if get_it:
            assert len(arg_list) == 1, 'Should be only one search parameter'
            def get_it_gen():
                def grouper(iterable, n, fillvalue=None):
                    args = [iter(iterable)] * n
                    return zip_longest(*args, fillvalue=fillvalue)
                for t, id, *cols in downsize_df[return_search_df()].itertuples():
                    for k in grouper([*cols], 4):
                        if arg_list[0] == 'subject':
                            if str(value_list[0]).lower() in str(k[0]).lower():
                                if get_it == 'predicate':
                                    yield id, k[1]
                                else:
                                    yield id, k[3]
                        elif arg_list[0] == 'predicate':
                            if str(value_list[2]).lower() in str(k[2]).lower():
                                if get_it == 'object':
                                    yield id, k[3]
                                else:
                                    yield id, k[1]
                        elif arg_list[0] == 'object':
                            if str(value_list[3]).lower() in str(k[3]).lower():
                                if get_it == 'subject':
                                    yield id, k[0]
                                else:
                                    yield id, k[1]

            df_data = [k for k in get_it_gen()]
            gen_df = pd.DataFrame(df_data, columns=['ID', 'Item'])

            gen_df.drop_duplicates(subset='Item',inplace=True)

            gen_df['Item'] = gen_df['Item'].apply(lambda x: re.sub(r'\n*','',x))
            return gen_df


        else:
            gen_df = extract_df[return_search_df()]
            gen_df['Notes'] = gen_df['Notes'].apply(lambda x: re.sub(r'\n*', '', x))
            return gen_df



    def discover(self, **kwargs):


        discovery_list = []

        ret_df = pd.DataFrame(columns = ['ID','Date','Time','Author','Type','Notes','Regx','Actions'])
        if 'sr' in kwargs.keys():
            self.db['regx_m'] = self.db.Notes.str.extract(r"([7|W]+\d{9})",expand=True)

            id_list = set(self.db.ID[self.db['regx_m'].notnull()])
            filtered_db = self.db[self.db['ID'].isin(id_list)]
            filtered_db = filtered_db[list(filtered_db.columns[:6]) + ['regx_m'] + [x for x in filtered_db.columns if (('predicate' in x) and ('lemma' not in x))]]

            filtered_db.sort_values(by=['ID','Date','Time'],ascending=[1,1,1],inplace=True)

            z = 0
            for i, id, date, time, author, type, notes, regx, *preds in filtered_db.itertuples():
                if id in id_list:
                    if str(regx) == 'nan':

                        discovery_list.append(','.join([x for x in preds if str(x) != 'nan']))
                    else:

                        ret_df = ret_df.append({'ID':id, 'Date':date, 'Time':time,
                                                'Author':author,'Type':type,'Notes':notes,
                                                'Regx': regx,'Actions':','.join(discovery_list)},ignore_index=True)
                        id_list.remove(id)
                        discovery_list = []
                        z+=1
                        continue

        return ret_df

y = TokenClassifier(r"path to file ", 'Notes') # here is where you specify the path
z = Explore(y.return_df())



z.discover(sr=True)
z.discover(sr=True).to_excel('test_example.xlsx')
