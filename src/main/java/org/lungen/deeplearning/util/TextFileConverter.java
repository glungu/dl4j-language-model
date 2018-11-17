package org.lungen.deeplearning.util;

import java.io.*;

/**
 * Created by user on 23.12.2017.
 */
public class TextFileConverter {

    public static void decodeFile(File source,
                                  String srcEncoding,
                                  File target,
                                  String targetEncoding) throws IOException {

        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(source), srcEncoding));
             BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
                     new FileOutputStream(target), targetEncoding));) {

            char[] buffer = new char[10000];
            int read;
            while ((read = br.read(buffer)) != -1) {
                bw.write(buffer, 0, read);
            }
        }
    }

    public static void appendFile(File sourceFile,
                                  String sourceEncoding,
                                  BufferedWriter writer) throws IOException {

        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(sourceFile), sourceEncoding))) {

            char[] buffer = new char[10000];
            int read;
            while ((read = br.read(buffer)) != -1) {
                writer.write(buffer, 0, read);
            }

            writer.write('\n');
            writer.flush();
        }
    }

    public static void main(String[] args) throws Exception {
        File dir = new File("C:/DATA/Projects/tolstoy/selected");
        File[] files = dir.listFiles(new FileFilter() {
            @Override
            public boolean accept(File f) {
                return f.getName().endsWith(".txt");
            }
        });
        try(BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
                new File(dir, "tolstoy_selected.txt")), "windows-1251"))) {

            for (File file : files) {
                appendFile(file, "windows-1251", bw);
                System.out.println("Appended: " + file.getName());
            }
        }

    }

}
