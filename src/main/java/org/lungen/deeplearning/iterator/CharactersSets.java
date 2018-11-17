package org.lungen.deeplearning.iterator;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * Utility class for obtaining Russian Character Set.
 */
public class CharactersSets {

    public static List<Character> RUSSIAN = getRussianCharacters();
    public static List<Character> LATIN = getLatinCharacters();
    public static List<Character> PUNCTUATION = getPunctuation();
    public static List<Character> NUMBERS = getNumbers();
    public static List<Character> SPECIAL = getSpecialSymbols();

    private static List<Character> getRussianCharacters() {
        List<Character> validChars = new LinkedList<>();
        for (char c = 'А'; c <= 'Я'; c++) validChars.add(c);
        for (char c = 'а'; c <= 'я'; c++) validChars.add(c);
        return validChars;
    }

    private static List<Character> getLatinCharacters() {
        List<Character> validChars = new LinkedList<>();
        for (char c = 'a'; c <= 'z'; c++) validChars.add(c);
        for (char c = 'A'; c <= 'Z'; c++) validChars.add(c);
        return validChars;
    }

    private static List<Character> getNumbers() {
        List<Character> validChars = new LinkedList<>();
        for (char c = '0'; c <= '9'; c++) validChars.add(c);
        return validChars;
    }

    private static List<Character> getPunctuation() {
        List<Character> validChars = new LinkedList<>();
        char[] chars = {
                '!', '&', '(', ')', '?',
                '-', '\'', '"', ',', '.',
                ':', ';', ' ', '\n', '\t'};
        for (char c : chars) validChars.add(c);
        return validChars;
    }

    private static List<Character> getSpecialSymbols() {
        List<Character> validChars = new LinkedList<>();
        char[] chars = {
                '@', '#', '$', '%', '^',
                '*', '{', '}', '[', ']',
                '/', '+', '_', '\\', '|',
                '<', '>'};
        for (char c : chars) validChars.add(c);
        return validChars;
    }

    public static char[] createCharacterSet(List<Character>... characterLists) {
        List<Character> chars = new ArrayList<>();
        for (List<Character> characterList : characterLists) {
            chars.addAll(characterList);
        }
        char[] validChars = new char[chars.size()];
        for (int i = 0; i < validChars.length; i++) {
            validChars[i] = chars.get(i);
        }
        return validChars;
    }

    public static char[] getRussianCharacterSet() {
        return createCharacterSet(RUSSIAN, LATIN, NUMBERS, PUNCTUATION);
    }

    public static char[] getEnglishCharacterSet() {
        return createCharacterSet(LATIN, NUMBERS, PUNCTUATION);
    }

//    /**
//     * A minimal character set:
//     * a-z, A-Z For Latin characters (e.g. French in War & Peace)
//     * а-я, А-Я For Russian characters
//     * 0-9 and common punctuation
//     */
//    public static char[] getMinimalCharacterSet() {
//        List<Character> validChars = new LinkedList<>();
//        for (char c = 'a'; c <= 'z'; c++) validChars.add(c);
//        for (char c = 'A'; c <= 'Z'; c++) validChars.add(c);
//        for (char c = '0'; c <= '9'; c++) validChars.add(c);
//        for (char c = 'А'; c <= 'Я'; c++) validChars.add(c);
//        for (char c = 'а'; c <= 'я'; c++) validChars.add(c);
//        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
//        for (char c : temp) validChars.add(c);
//        char[] out = new char[validChars.size()];
//        int i = 0;
//        for (Character c : validChars) out[i++] = c;
//        return out;
//    }
//
//    /** As per getMinimalCharacterSet(), but with a few extra characters */
//    public static char[] getDefaultCharacterSet() {
//        List<Character> validChars = new LinkedList<>();
//        for (char c : getMinimalCharacterSet()) validChars.add(c);
//        char[] additionalChars = {'@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
//                '\\', '|', '<', '>'};
//        for (char c : additionalChars) validChars.add(c);
//        char[] out = new char[validChars.size()];
//        int i = 0;
//        for (Character c : validChars) out[i++] = c;
//        return out;
//    }

}
