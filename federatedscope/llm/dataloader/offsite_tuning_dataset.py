import re
import datasets


class PIQA:
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

        self.dataset = datasets.load_dataset("piqa")

    def get_context(self, examples):
        ctx = examples['goal']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        if -1 in examples["label"]:  # test set
            return [""] * len(examples["label"])
        else:
            gt_tuples = [("sol{}".format(label + 1), idx)
                         for idx, label in enumerate(examples['label'])]
            return [examples[k][i] for k, i in gt_tuples]

    def get_data_dict(self, label='train'):
        contexts = self.get_context(self.dataset[label])
        targets = self.get_target(self.dataset[label])

        return [
            dict(context=context, target=target, category='piqa')
            for (context, target) in zip(contexts, targets)
        ]


class HellaSwag:
    def __init__(self):
        # Download datasets
        self.dataset = datasets.load_dataset('Rowan/hellaswag')

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow
        # dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def get_context(self, examples):
        ctx_zip = zip(examples["activity_label"], examples["ctx_a"],
                      examples["ctx_b"])
        return [
            self.preprocess(a + ": " + b + " " + c.capitalize())
            for a, b, c in ctx_zip
        ]

    def get_target(self, examples):
        labels = examples["label"]
        endings = examples["endings"]
        targets = []
        for idx, label in enumerate(labels):
            target = '' if label == '' else endings[idx][int(label)]
            targets.append(self.preprocess(target))
        return targets

    def get_data_dict(self, label='train'):
        contexts = self.get_context(self.dataset[label])
        targets = self.get_target(self.dataset[label])

        return [
            dict(context=context, target=target, category='hellaswag')
            for (context, target) in zip(contexts, targets)
        ]


class OpenBookQA:
    def __init__(self):
        # Download datasets
        self.dataset = datasets.load_dataset('openbookqa')

    def get_context(self, examples):
        return examples['question_stem']

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        targets = []
        for choice, answer in zip(choices, answers):
            answer = ord(answer.strip()) - ord('A')
            targets.append(choice['text'][answer])
        return targets

    def get_data_dict(self, label='train'):
        contexts = self.get_context(self.dataset[label])
        targets = self.get_target(self.dataset[label])

        return [
            dict(context=context, target=target, category='openbookqa')
            for (context, target) in zip(contexts, targets)
        ]


class ARC:
    def __init__(self, name):
        self._template = "Question: {}\nAnswer:"

        # Download datasets
        assert name in ['ARC-Challenge', 'ARC-Easy']
        self.name = name
        self.dataset = datasets.load_dataset('ai2_arc', name)

    def get_context(self, examples):
        ctx = examples['question']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        for idx, answer in enumerate(answers):
            answer = num_to_letter.get(answer, answer)
            answer = ord(answer) - ord("A")
            answers[idx] = choices[idx]["text"][answer]
        return answers

    def get_data_dict(self, label='train'):
        contexts = self.get_context(self.dataset[label])
        targets = self.get_target(self.dataset[label])

        return [
            dict(context=context, target=target, category=self.name)
            for (context, target) in zip(contexts, targets)
        ]


class RACE:
    def __init__(self):
        # Download datasets
        self.dataset = datasets.load_dataset('race', 'high')

    @classmethod
    def doc_to_text(cls, article, question):
        text = "Article: " + article + "\n\n"
        text += "Question: " + question + "\n\n"
        text += "Answer:"
        return text

    def get_context(self, examples):
        return [
            self.doc_to_text(article, question) for article, question in zip(
                examples["article"], examples["question"])
        ]

    def get_target(self, examples):
        answers = examples['answer']
        options = examples['options']
        for idx, answer in enumerate(answers):
            answers[idx] = options[idx][ord(answer) - ord("A")]
        return answers

    def get_data_dict(self, label='train'):
        contexts = self.get_context(self.dataset[label])
        targets = self.get_target(self.dataset[label])

        return [
            dict(context=context, target=target, category='race')
            for (context, target) in zip(contexts, targets)
        ]


class SciQ:
    def __init__(self):
        self._template = "{}\nQuestion: {}\nAnswer:"

        # Download datasets
        self.dataset = datasets.load_dataset('sciq')

    def get_context(self, examples):
        sources = examples['support']
        queries = examples['question']
        return [self._template.format(s, q) for s, q in zip(sources, queries)]

    def get_target(self, examples):
        return examples['correct_answer']

    def get_data_dict(self, label='train'):
        contexts = self.get_context(self.dataset[label])
        targets = self.get_target(self.dataset[label])

        return [
            dict(context=context, target=target, category='sciq')
            for (context, target) in zip(contexts, targets)
        ]


class WebQs:
    def __init__(self):
        # Download datasets
        self.dataset = datasets.load_dataset('web_questions')

    def get_context(self, examples):
        return [
            "Question: " + question + "\nAnswer:"
            for question in examples["question"]
        ]

    def get_target(self, examples):
        return [" " + answers[0] for answers in examples["answers"]]

    def get_data_dict(self, label='train'):
        contexts = self.get_context(self.dataset[label])
        targets = self.get_target(self.dataset[label])

        return [
            dict(context=context, target=target, category='webqs')
            for (context, target) in zip(contexts, targets)
        ]


# task_dict = {
#     "piqa": PIQA(),
#     "hellaswag": HellaSwag(),
#     "openbookqa": OpenBookQA(),
#     "arc_easy": ARC(),
#     "arc_challenge": ARC(),
#     "sciq": SciQ(),
#     "web_questions": WebQs(),
#     "race": RACE(),
# }

# def map_dataset_name_and_config(args):
#     dataset_name = args.dataset_name
#     dataset_config_name = args.dataset_config_name
#     if args.dataset_name == 'arc_easy':
#         dataset_name = 'ai2_arc'
#         dataset_config_name = 'ARC-Easy'
#     elif args.dataset_name == 'arc_challenge':
#         dataset_name = 'ai2_arc'
#         dataset_config_name = 'ARC-Challenge'
#     elif args.dataset_name == 'race':
#         dataset_config_name = 'high'

#     return dataset_name, dataset_config_name

# LM_EVAL_TASK_NAME_MAPPING = {
#     "web_questions": "webqs"
# }

if __name__ == "__main__":
    PIQA().get_dataset()
