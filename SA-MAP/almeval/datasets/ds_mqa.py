# Modified from https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/master/almeval/datasets/ds_mqa.py

import pandas as pd
from .base import AudioBaseDataset


class AudioMQADataset(AudioBaseDataset):
    INTERACTIVE = 'Audio-analysis'
    TASK = 'MQA'
    LANG = None

    def evaluate(self, eval_file, dump_judge=True, method='default'):
        if method == 'vb-mcq':
            if "Qwen2-Audio-7B" in eval_file:
                metrics, judge_results = self.evaluate_vb_mcq(
                    eval_file)
            else:
                metrics, judge_results = self.evaluate_vb_mcq_kimi(
                    eval_file)
            judge_model_name = 'vb-mcq'
        else:
            NotImplementedError

        model_name = self.get_model_name(eval_file)
        result = self.format_performance(
            model_name, metrics, eval_method=method)

        if dump_judge:
            # dump the judge result to the eval_file
            all_df = []
            for task, judge_result in judge_results.items():
                df = pd.DataFrame(judge_result)
                all_df.append(df)
            all_df = pd.concat(all_df)
            save_file = eval_file.replace(
                '.jsonl', f'_{judge_model_name}_judge.jsonl')
            all_df.to_json(save_file, orient='records',
                           lines=True, force_ascii=False)
        return result

    def extract_answer_vb_mcq(self, response, choices):
        response = response.lower()
        if response.startswith('<1>') or response.startswith('<2>') or response.startswith('<3>'):
            response = response[3:].strip()
        for template in [
            '答案是[CHOICE]',
            '答案是 [CHOICE]',
            '答案是选项[CHOICE]',
            '答案应该是[CHOICE]',
            '答案应该是 [CHOICE]',
            '答案就是选项[CHOICE]',
            '答案是‘[CHOICE]',
            '是[CHOICE]：',
            '答案选[CHOICE]',
            '[CHOICE]是正确',
            '选项[CHOICE]是最合适的',
            'answer is: **[CHOICE]',
            'answer is **[CHOICE]',
            'the answer to the question is: **[CHOICE]',
            'the answer to the multiple-choice question is **[CHOICE]',
            "the answer is '[CHOICE]'",
            '[CHOICE] is the best answer',
            'the answer is [CHOICE]',
            'the correct answer is [CHOICE]',
            'would select [CHOICE]',
            'would choose [CHOICE]',
            'would select option [CHOICE]',
            'would choose option [CHOICE]',
            'is \"[CHOICE]\"',
            'is \"[CHOICE].',
            'is: **[CHOICE])',
            'is **[CHOICE],',
            'is **[CHOICE]:',
            'is **[CHOICE])',
            'is: **[CHOICE].',
            'is: **[CHOICE]:',
            'is **[CHOICE].',
            'be **[CHOICE],',
            'is: **[CHOICE]**',
            'is therefore option **[CHOICE]:',
            'is: \n\n**[CHOICE])',
            'as **[CHOICE]:',
            'be **[CHOICE])',
            'be **[CHOICE]:',
            'is: \n\n**[CHOICE]**',
            'suggests **[CHOICE])',
            'be option **[CHOICE]:',
            'with **[CHOICE])',
            "is typically \"[CHOICE])",
            'be to **[CHOICE])',
            'is: \n\n[CHOICE])',
            'is likely to be: **[CHOICE].',
            'is **[CHOICE] (',
            'is option **[CHOICE]**',
            'is likely **[CHOICE]**',
            'is:\n**[CHOICE].',
            'is:\n\n**[CHOICE].',
            'would be [CHOICE]',
            'would be option [CHOICE]',
            'would be ([CHOICE])',
            'would be option ([CHOICE])',
            'is [CHOICE],',
            'is typically [CHOICE],',
            'is typically [CHOICE].',
            "i'd say [CHOICE].",
            'option [CHOICE].',
            'option [CHOICE]:',
            'option [CHOICE],',
            'the answer is:\n**[CHOICE]',
            'is [CHOICE]:',
            'is [CHOICE].',
            'is [CHOICE],',
            'is: [CHOICE].',
            'is ([CHOICE])',
            'is:\n**[CHOICE])',
            'is likely **[CHOICE]:',
            'is the **[CHOICE])',
            ':\n[CHOICE].',
            ':\n[CHOICE])',
            ':\n[CHOICE],',
            ': \n[CHOICE].',
            ':  \n[CHOICE].',
            ':\n\n[CHOICE].',
            ':\n\n[CHOICE])',
            'is most likely **[CHOICE]:',
            ':\n\n[CHOICE],',
            ': \n\n[CHOICE].',
            'is option [CHOICE],',
            '([CHOICE]) would be',
            'is ([CHOICE]).',
            'is [CHOICE])',
            'is: [CHOICE])',
            'is:\n\n[CHOICE]:',
            'is: **[CHOICE],',
            '(option [CHOICE])',
            'answer is ([CHOICE])',
            "select option \"[CHOICE]\"",
            'is: [CHOICE]',
            'is typically **[CHOICE],',
            'is **[CHOICE]**',
            "is likely '[CHOICE]'",
            "is option '[CHOICE]'",
            'is:\n**[CHOICE]:',
            'is \\( \\boxed{[CHOICE] ',
            "would be '[CHOICE]'",
            'is the **[CHOICE]** ',
            'question is [CHOICE] (',
            'is:\n\n**[CHOICE])',
            'closest to option **[CHOICE]**',
            'is most likely **[CHOICE])',
            "the answer to the question is '[CHOICE]'",
            'question is **[CHOICE]**',
            "known as '[CHOICE]'",
            "is '[CHOICE])",
            'is typically **[CHOICE]:',
            'is \\( \\boxed{\\text{[CHOICE]}} \\)',
            'is \\( \\text{[CHOICE]) }',
            'is \\( \\text{[CHOICE]} \\)',
            'is \\( \\text{[CHOICE]:',
            'is \\( \\text{[CHOICE])',
            'is \\(\\text{[CHOICE].',
            'is:\n\n**[CHOICE]',
            'is \\( \\text{[CHOICE].}',
            'is \\( \\text{[CHOICE].',
            'is \\( \\boxed{[CHOICE]}',
            'is:\n\\[ \\boxed{\\text{[CHOICE]}}',
            'is:\n\\[ \\text{[CHOICE])',
            'is:\n\n\\[ \\text{[CHOICE])',
            'is \\( \\textbf{[CHOICE])',
            'is \\( \\text{[CHOICE]}',
            'is: \\( \\text{[CHOICE].',
            'corresponds to:\n- **[CHOICE]:',
            'would be: **[CHOICE]**.',
            'is \\( [CHOICE] \\)',
            'is:\n**[CHOICE] ',
            'corresponds to option **[CHOICE]**',
            'be **[CHOICE]**',
            'be: \n\n[CHOICE])',
            'is:\n\\[ \\boxed{[CHOICE]}',
            'is:  \n**[CHOICE]:',
            'is: \\( \\text{[CHOICE])',
            'is likely: **[CHOICE],',
            'is } \\mathbf{[CHOICE].',
            'is \\( \\boxed{[CHOICE])',
            'is \\( \\textbf{[CHOICE]}',
            'is \\([CHOICE]\\)',
            'is:\n  \n**[CHOICE]:',
            'is option **[CHOICE] ',
            'is:\n\\( \\textbf{[CHOICE].',
            'is \\( \\mathbf{[CHOICE]}',
            'was option **[CHOICE]**',
            "is likely \"[CHOICE])",
            'option **[CHOICE]:',
            "is \"[CHOICE])",
            'is most likely **[CHOICE],',
            'is often **[CHOICE]:',
            'is:  \n[CHOICE])',
            ' [CHOICE].',
            ' [CHOICE],',
            ' [CHOICE]:',
            ' [CHOICE])',
            '**[CHOICE].',
            '**[CHOICE])',
            "\"[CHOICE].",
            "\"[CHOICE],",
            "\"[CHOICE]:",
            '([CHOICE])',
            "\"[CHOICE]\"",

        ]:
            for choice in choices:
                if template.replace('[CHOICE]', choice) in response:
                    return choice.upper()
        for choice in choices:
            if response == choice:
                return choice.upper()
            if response.strip() == choice:
                return choice.upper()
            for punc in ['.', ',', ':', ')']:
                if response.startswith(choice+punc):
                    return choice.upper()

        if 'would be a.' in response:
            return 'A'
        elif 'would be \"a.' in response:
            return 'A'
        elif 'the best option from the given choices would be a scorpion (a)' in response:
            return 'A'
        else:
            return response

    def evaluate_vb_mcq(self, eval_file):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        for task, group in df.groupby('subset'):
            if task == 'sentiment':
                continue
            options = [self.extract_question_vb_mcq(prompt) for prompt in group['real_prompt']]
            choices = [[letter for letters in option.values() for letter in letters] for option in options]
            ground_truth = group['answer'].astype(str).to_list()
            ground_truth_new = [options[i][truth.strip().lower()] for i, truth in enumerate(ground_truth)]
            preds = group['prediction'].astype(str).to_list()
            preds_new = [self.extract_answer_vb_mcq(pred, choices[i]) for i, pred in enumerate(preds)]

            results = []
            for idx, (pred, gt, pred_new, gt_new) in enumerate(zip(preds, ground_truth, preds_new, ground_truth_new)):
                if pred is None:
                    results.append((idx, None))
                elif pred_new.lower() in gt_new:
                    results.append((idx, 'yes'))
                elif gt in pred:
                    results.append((idx, 'yes'))
                else:
                    results.append((idx, 'no'))
            task_result, judge_result = self.collect_acc(results, group)
            metrics[task] = task_result
            print(f'{task} result: {task_result}')
            judge_results[task] = judge_result
        return metrics, judge_results
    
    def evaluate_vb_mcq_kimi(self, eval_file):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        for task, group in df.groupby('subset'):
            if task == 'sentiment':
                continue
            options = [self.extract_question_vb_mcq(prompt) for prompt in group['real_prompt']]
            choices = [[letter for letters in option.values() for letter in letters] for option in options]
            ground_truth = group['answer'].astype(str).to_list()
            ground_truth_new = [options[i][truth.strip().lower()] for i, truth in enumerate(ground_truth)]
            preds = group['prediction'].astype(str).to_list()
            preds_new = [self.extract_answer_vb_mcq(pred, choices[i]) for i, pred in enumerate(preds)]
            results = []
            for idx, (pred, gt, pred_new, gt_new) in enumerate(zip(preds, ground_truth, preds_new, ground_truth_new)):
                if pred is None:
                    results.append((idx, None))
                elif pred_new.lower() in gt_new:
                    results.append((idx, 'yes'))
                else:
                    results.append((idx, 'no'))
            task_result, judge_result = self.collect_acc(results, group)
            metrics[task] = task_result
            print(f'{task} result: {task_result}')
            judge_results[task] = judge_result
        return metrics, judge_results
    
    def extract_question_vb_mcq(self, question_text):
        import re
        stop_markers = ["Your answer is:"]
        question_part = question_text
        for marker in stop_markers:
            if marker in question_text:
                question_part = question_text.split(marker)[0]
                break
        option_pattern = r'\(([A-Z])\)\s*((?:(?!\s*\([A-Z]\)).)*)'
        matches = re.findall(option_pattern, question_part)
        options = {}
        for letter, text in matches:
            cleaned_text = text.strip().lower()
            if cleaned_text not in options:
                options[cleaned_text] = []
            options[cleaned_text].append(letter.lower())
        return options


class MMAUTestMini(AudioMQADataset):
    DATASET_NAME = 'mmau-test-mini'
    DATASET_SERIES = 'MMAU'
    AUDIO_TYPE = 'AudioEvent'


class MELD(AudioMQADataset):
    DATASET_NAME = 'MELD'
    DATASET_SERIES = 'MELD'
    AUDIO_TYPE = 'Speech'

    def evaluate(self, eval_file, dump_judge=True, method='default'):
        if method == 'vb-mcq':
            if "Qwen2-Audio-7B" in eval_file:
                metrics, judge_results = self.evaluate_vb_mcq_qwen2_audio(
                    eval_file)
            else:
                metrics, judge_results = self.evaluate_vb_mcq(
                    eval_file)
            judge_model_name = 'vb-mcq'
        else:
            NotImplementedError

        model_name = self.get_model_name(eval_file)
        result = self.format_performance(
            model_name, metrics, eval_method=method)

        if dump_judge:
            # dump the judge result to the eval_file
            all_df = []
            for task, judge_result in judge_results.items():
                df = pd.DataFrame(judge_result)
                all_df.append(df)
            all_df = pd.concat(all_df)
            save_file = eval_file.replace(
                '.jsonl', f'_{judge_model_name}_judge.jsonl')
            all_df.to_json(save_file, orient='records',
                           lines=True, force_ascii=False)
        return result
    
    def is_correct(self, truth, prediction):
        synonyms = {
            'neutral': ['neutral', 'calm', 'normal', 'indifferent', 'unemotional'],
            'joy': ['joy', 'happy', 'happiness', 'glad', 'pleased', 'delighted', 'joyful'],
            'happy': ['joy', 'happiness', 'glad', 'pleased', 'delighted'],
            'sadness': ['sadness', 'sad', 'unhappy', 'depressed', 'sorrowful', 'gloomy'],
            'sad': ['sadness', 'unhappy', 'depressed', 'sorrowful'],
            'anger': ['anger', 'angry', 'mad', 'furious', 'irritated', 'annoyed'],
            'angry': ['anger', 'mad', 'furious', 'irritated'],
            'surprise': ['surprise', 'surprised', 'astonished', 'amazed', 'shocked', 'startled'],
            'surprised': ['surprise', 'astonished', 'amazed', 'shocked'],
            'fear': ['fear', 'afraid', 'scared', 'frightened', 'terrified', 'anxious'],
            'afraid': ['fear', 'scared', 'frightened'],
            'disgust': ['disgust', 'disgusted', 'revulsion', 'repulsed', 'nauseated'],
            'disgusted': ['disgust', 'revulsion', 'repulsed']
        }
        if truth.lower() == prediction.lower():
            return True
        truth_key = truth.lower()
        prediction_val = prediction.lower()
        if truth_key in synonyms:
            if prediction_val in synonyms[truth_key]:
                return True
        for key, syn_list in synonyms.items():
            if prediction_val == key and truth_key in syn_list:
                return True
            if prediction_val in syn_list and truth_key == key:
                return True
        
        return False

    def evaluate_vb_mcq_qwen2_audio(self, eval_file):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        for task, group in df.groupby('subset'):
            if task == 'sentiment':
                continue
            ground_truth = group['answer'].astype(str).to_list()
            preds = group['prediction'].astype(str).to_list()
            preds = [pred.lstrip() for pred in preds]
            results = []
            for idx, (pred, gt) in enumerate(zip(preds, ground_truth)):
                if pred is None:
                    results.append((idx, None))
                elif self.is_correct(gt, pred):
                    results.append((idx, 'yes'))
                else:
                    results.append((idx, 'no'))
            task_result, judge_result = self.collect_acc(results, group)
            metrics[task] = task_result
            print(f'{task} result: {task_result}')
            judge_results[task] = judge_result
        return metrics, judge_results


class Nonspeech7k(AudioMQADataset):
    DATASET_NAME = 'Nonspeech7k'
    DATASET_SERIES = 'Nonspeech7k'
    AUDIO_TYPE = 'AudioEvent'


class TUT2017(AudioMQADataset):
    DATASET_NAME = 'TUT2017'
    DATASET_SERIES = 'TUT2017'
    AUDIO_TYPE = 'AudioEvent'


class Vocalsound(AudioMQADataset):
    DATASET_NAME = 'VocalSound'
    DATASET_SERIES = 'VocalSound'
    AUDIO_TYPE = 'AudioEvent'
