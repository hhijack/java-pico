package com.example;

import java.util.HashMap;
import java.util.Map;

public class UserService {
    private Map<String, User> userStore = new HashMap<>();

    public User findById(String id) {
        return userStore.get(id);
    }

    public void save(User user) {
        userStore.put(user.getId(), user);
    }

    public boolean exists(String id) {
        return userStore.containsKey(id);
    }

    // find all users
    public Map<String, User> findAll() {
        return userStore;
    }
}

class User {
    private String id;
    private String name;
    private String email;

    public User(String id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
    }

    public String getId() { return id; }
    public String getName() { return name; }
    public String getEmail() { return email; }
}
