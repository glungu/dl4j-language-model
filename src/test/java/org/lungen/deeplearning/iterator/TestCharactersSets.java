package org.lungen.deeplearning.iterator;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.stream.Stream;

/**
 * TestCharactersSets
 *
 * @author lungen.tech@gmail.com
 */
public class TestCharactersSets {

    @Test
    public void testRussianLowerCase() {
        Character[] chars = new Character[] {
                'а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о',
                'п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я'
        };
        Stream.of(chars).forEach(c -> {
            Assert.assertTrue("Missing letter: " + c,
                    CharactersSets.RUSSIAN_LOWERCASE.contains(c));
        });
    }
}
