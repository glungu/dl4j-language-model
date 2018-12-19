package org.lungen.deeplearning.listener;

import java.io.IOException;

import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * StatsListener that reports stats to memory stats store,
 * and attaches it to UI Server.
 *
 */
public class UIStatsListener extends StatsListener {

    private static final Logger log = LoggerFactory.getLogger("listener.ui");

    public UIStatsListener() {
        super(new InMemoryStatsStorage());
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach((InMemoryStatsStorage) getStorageRouter());
    }

    public void close() {
        InMemoryStatsStorage statsStorage = (InMemoryStatsStorage) getStorageRouter();
        try {
            statsStorage.close();
        } catch (IOException e) {
            log.error("Cannot close stats storage", e);
        }
    }

}
