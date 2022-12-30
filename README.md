# Fake News Detection using Deep Learning

This project proposed Word2Vec-LSTM methods for detecting fake news in Indonesian text language. As many as 4,800 news data from three Indonesian news sites with high credibility (cnnindonesia, detik.co.id, and turnbackhoax.id) are obtained through the crawl/scrap data process and used for this learning project. The average of the 10-fold cross-validation score results shows that:

* Accuracy: 89.42%
* Precision: 88.76%
* Recall: 92.53%
* F1 Score: 89.97%

## Model

link : https://drive.google.com/file/d/1D22KQp4lMTtF4EamzuDkR86lGUbg5seX/

## Usage

1. Clone this repository.
2. Download the model from gdrive link.
3. Run `app.py`, go to the `http://127.0.0.1:5000/`
4. Paste news text in text area input and submit.

- The UI design template reference is by Pooja Bhagat: [https://github.com/581-pooja ](https://github.com/581-pooja/Fake-News-Classification-App/tree/master/templates)

## Note
- This project is only for learning purposes and doesn't work well in real time because was trained on historical and limited data.


## Paper of Project
<a id="1">[1]</a> 
R. Yusuf and S. Suyanto, "Hoax Detection on Indonesian Text using Long Short-Term Memory," 2022 5th International Conference on Information and Communications Technology (ICOIACT), 2022, pp. 268-271, doi: 10.1109/ICOIACT55506.2022.9972086.
