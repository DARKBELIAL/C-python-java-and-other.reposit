import java.util.ArrayList;
import java.util.Scanner;

class Book {
    private String title;
    private String author;
    private boolean available;

    public Book(String title, String author) {
        this.title = title;
        this.author = author;
        this.available = true;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public boolean isAvailable() {
        return available;
    }

    public void borrow() {
        available = false;
    }

    public void returnBook() {
        available = true;
    }

    @Override
    public String toString() {
        return "Title: " + title + "\nAuthor: " + author + "\nAvailable: " + (available ? "Yes" : "No");
    }
}

class Library {
    private ArrayList<Book> books = new ArrayList<>();

    public void addBook(Book book) {
        books.add(book);
    }

    public void displayBooks() {
        System.out.println("Books in the Library:");
        for (Book book : books) {
            System.out.println(book);
        }
    }

    public void borrowBook(String title) {
        for (Book book : books) {
            if (book.getTitle().equalsIgnoreCase(title) && book.isAvailable()) {
                book.borrow();
                System.out.println("You have borrowed the book: " + title);
                return;
            }
        }
        System.out.println("Sorry, the book is not available for borrowing.");
    }

    public void returnBook(String title) {
        for (Book book : books) {
            if (book.getTitle().equalsIgnoreCase(title) && !book.isAvailable()) {
                book.returnBook();
                System.out.println("You have returned the book: " + title);
                return;
            }
        }
        System.out.println("You cannot return this book as it was not borrowed from the library.");
    }
}

public class LibraryManagementSystem {
    public static void main(String[] args) {
        Library library = new Library();

        Book book1 = new Book("Introduction to Java", "John Smith");
        Book book2 = new Book("Data Structures and Algorithms", "Alice Johnson");
        Book book3 = new Book("The Art of Programming", "Bob Davis");

        library.addBook(book1);
        library.addBook(book2);
        library.addBook(book3);

        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("\nLibrary Management System");
            System.out.println("1. Display Books");
            System.out.println("2. Borrow a Book");
            System.out.println("3. Return a Book");
            System.out.println("4. Exit");
            System.out.print("Enter your choice: ");

            int choice = scanner.nextInt();
            scanner.nextLine(); // Consume newline

            switch (choice) {
                case 1:
                    library.displayBooks();
                    break;
                case 2:
                    System.out.print("Enter the title of the book you want to borrow: ");
                    String borrowTitle = scanner.nextLine();
                    library.borrowBook(borrowTitle);
                    break;
                case 3:
                    System.out.print("Enter the title of the book you want to return: ");
                    String returnTitle = scanner.nextLine();
                    library.returnBook(returnTitle);
                    break;
                case 4:
                    System.out.println("Thank you for using the Library Management System. Goodbye!");
                    scanner.close();
                    System.exit(0);
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        }
    }
}
