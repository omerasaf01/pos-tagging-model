from main import pos_tag


def main():
    while True:
        text = input("Text: ")
        print(pos_tag(text)._ents)


if __name__ == "__main__":
    main()
