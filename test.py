from src.main import ISLRecognitionApp


def main():
    app = ISLRecognitionApp()
    print("App initialized successfully.")
    print(f"Loaded classes: {list(app.model.classes_)}")
    print(f"Feature count: {app.model.n_features_in_}")


if __name__ == "__main__":
    main()
