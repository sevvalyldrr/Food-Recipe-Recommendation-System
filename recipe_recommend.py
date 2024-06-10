import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

turkish_chars = {"I": "ı", "İ": "i", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö"}


def load_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data


def preprocess_data(data):
    recipes = []
    for recipe_id, recipe_data in data["Recipe"].items():
        category = recipe_data["CategoryBread"]
        ingredients = recipe_data["IngridientNames"]
        name = recipe_data["Name"]
        ingredients = [
            "".join(turkish_chars.get(c, c) for c in ingredient.strip()).lower()
            for ingredient in ingredients.split(";")
            if ingredient.strip()
        ]
        ingredients = ", ".join(ingredients)
        recipes.append({"Category": category, "Ingredients": ingredients, "Name": name})
    df = pd.DataFrame(recipes)
    return df


def build_model():
    random_forest = RandomForestClassifier(
        class_weight="balanced",
        criterion="gini",
        n_estimators=287,
        max_depth=42,
        min_samples_split=5,
        max_leaf_nodes=None,
        random_state=50,
    )
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    pipeline = Pipeline([("tfidf", tfidf_vectorizer), ("clf", random_forest)])
    return pipeline


def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    print("Accuracy:", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j + 0.5, i + 0.5, conf_matrix[i, j], ha="center", va="center")
    sns.heatmap(conf_matrix, annot=True, fmt=".0f")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title("Accuracy Score: {0}".format(accuracy), size=15)
    plt.show()

    return accuracy


def plot_performance(X_train, y_train, X_test, y_test):
    # Modelin eğitim ve test setlerindeki performansını izleme
    train_accuracy = []
    test_accuracy = []
    for i in range(10, 310, 10):  # Modelin 10'ar ağaç ekleyerek eğitimini gözlemleyelim
        # Random Forest sınıflandırıcı ve parametreler
        random_forest = RandomForestClassifier(
            class_weight='balanced',
            criterion='gini',
            n_estimators=i,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=1,
            max_leaf_nodes=None,
            random_state=50
        )

        # Pipeline oluşturma
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=300)), ('clf', random_forest)
        ])

        # Modeli eğitme
        pipeline.fit(X_train, y_train)

        # Eğitim seti üzerinde doğruluk
        train_pred = pipeline.predict(X_train)
        train_accuracy.append(accuracy_score(y_train, train_pred))

        # Test seti üzerinde doğruluk
        test_pred = pipeline.predict(X_test)
        test_accuracy.append(accuracy_score(y_test, test_pred))

    plt.plot(range(10, 310, 10), train_accuracy, label='Training Accuracy')
    plt.plot(range(10, 310, 10), test_accuracy, label='Test Accuracy')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs. Number of Trees')
    plt.legend()
    plt.show()


def predict_category(pipeline, ingredients):
    ingredients = [
        "".join(turkish_chars.get(c, c) for c in ingredient.strip()).lower()
        for ingredient in ingredients.split(",")
        if ingredient.strip()
    ]
    ingredients = ', '.join(ingredients)
    prediction = pipeline.predict([ingredients])
    return prediction[0]


def get_recipes_by_category(df, category):
    recipes = df[df["Category"] == category]
    return recipes


def main():
    # Veri yükleme ve ön işleme
    data = load_data("dataset.json")
    df = preprocess_data(data)

    # Model oluşturma ve eğitim
    pipeline = build_model()
    X_train, X_test, y_train, y_test = train_test_split(df["Ingredients"], df["Category"], test_size=0.17, random_state=27)
    train_model(pipeline, X_train, y_train)

    # Model değerlendirme
    evaluate_model(pipeline, X_test, y_test)

    # Model performansını çizme
    plot_performance(X_train, y_train, X_test, y_test)

    # Kullanıcıdan malzemeleri alma ve tahmin yapma döngüsü
    while True:
        ingredients = input("Lütfen en az üç malzeme girin (malzemeleri ',' ile ayırın) veya çıkış yapmak için 'q' yazın: ")
        if ingredients.lower() == 'q':
            break
        prediction = predict_category(pipeline, ingredients)
        print("Girilen malzemelerle tahmin edilen yemek kategorisi:", prediction)

        show_recipes = input(f"{prediction} kategorisindeki tüm tarifleri görmek ister misiniz? (e/h): ")
        if show_recipes.lower() == 'e':
            recipes = get_recipes_by_category(df, prediction)
            for idx, row in recipes.iterrows():
                print(f"{row['Name']}")

            # Kullanıcının girdiği tarif
            recipe_query = input("Lütfen aramak istediğiniz yemek tarifinin adını girin: ")

            # WebDriver'ı başlatma
            driver = webdriver.Chrome()

            try:
                driver.get("https://yemek.com/ara/")  # Arama yapılacak web sitesinin URL'si
                print("Siteye gidildi.")

                 # Arama kutusunu bulma ve tarif arama
                wait = WebDriverWait(driver, 20)  # Bekleme süresi 20 saniye olarak ayarlandı

                # CSS Selector kullanarak arama kutusunu bulma (güncellenmiş seçici)
                try:
                    search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'][placeholder='Tarif, yiyecek, içecek, ipucu ara...']")))
                except:
                    print("CSS Selector ile arama kutusu bulunamadı.")
                    raise

                print("Arama kutusu bulundu.")
                search_box.send_keys(recipe_query)
                search_box.send_keys(Keys.RETURN)
                print("Arama yapıldı.")

                 # Arama sonuçlarını beklemek için zaman tanıma
                time.sleep(10)  # Sayfanın yüklenmesi için bekleme süresi

                # İlk arama sonucunu bulma ve tıklama (güncellenmiş seçici)
                try:
                    first_result = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.recipe-card__title-link")))
                except:
                    print("CSS Selector ile ilk sonuç bulunamadı.")
                    raise

                print("İlk arama sonucu bulundu.")
                first_result.click()
                print("İlk arama sonucuna tıklandı.")

                # Sayfanın yüklenmesini bekleme
                time.sleep(10)

                # Tarayıcıyı kapatmamak için sonsuz döngü (isteğe bağlı, sayfayı görmek için)
                while True:
                    pass

            except Exception as e:
                print(f"Bir hata oluştu: {e}")

            finally:
                # Tarayıcıyı kapatma
                driver.quit()
                print("Tarayıcı kapatıldı.")


if __name__ == "__main__":
    main()
