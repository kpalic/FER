package hr.fer.progi.simplicity.services;

import hr.fer.progi.simplicity.entities.User;

import java.util.List;
import java.util.Map;

public interface MapService {
    Map<String, List<?>> getMapInfo();
    Map<String, List<?>> getMapUserInfo(User user);
}
