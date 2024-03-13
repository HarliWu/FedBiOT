from transformers import AutoTokenizer, AutoModelForCausalLM

auto_j_model = AutoModelForCausalLM.from_pretrained("GAIR/autoj-13b")
auto_j_tokenizer = AutoTokenizer.from_pretrained("GAIR/autoj-13b")


def read_file(path='test_results.txt'):
    f = open(path, 'r', encoding="utf8")
    colletion = []

    tag = ''
    record = {'subreddit': '', 'title': '', 'post': '', 'response': ''}
    for line in f.readlines():
        if 'Subreddit: ' in line:
            record['subreddit'] = line.replace('Subreddit: ', '')

        if 'Title: ' in line:
            tag = 'title'

        if 'Post:' in line:
            tag = 'post'

        elif 'Best generated summary' in line:
            tag = 'response'

        elif '=============' in line:
            # The end of the record, which should be
            # appended to the collection
            query = ("Write a summary for a reddit post.\n\n"
                     "Title: {title}\n\n"
                     "Post: {post}\n\n").format_map(record)
            record['query'] = query
            colletion.append(record)
            record = {'subreddit': '', 'title': '', 'post': '', 'response': ''}

        else:
            # This is a normal line and should be
            # extended to current tag of record
            record[tag] = record[tag] + line

    return colletion


def extract_single_rating(score_output):
    pred_score = 0.0
    if "Rating: [[" in score_output:
        pos = score_output.rfind("Rating: [[")
        pos2 = score_output.find("]]", pos)
        assert pos != -1 and pos2 != -1
        pred_score = float(score_output[pos + len("Rating: [["):pos2].strip())
    return pred_score


def auto_j_eval_rating(dataset):
    prompt = (
        "Write critiques for a submitted response on a given user's query, "
        "and grade the response:\n\n"
        "[BEGIN DATA]\n***\n[Query]: {query}\n***\n"
        "[Response]: {response}\n***\n[END DATA]\n\n"
        "Write critiques for this response. After that, you should "
        "give a final rating for the response on a scale of 1 to 10 "
        "by strictly following this format: \"[[rating]]\", "
        "for example: \"Rating: [[5]]\".")
