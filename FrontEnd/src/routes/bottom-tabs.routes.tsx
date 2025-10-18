import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import Home from "../screens/Home/Home";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { ScanStack } from "../screens/Scan/Scan";
import { View } from "react-native";

const Tab = createBottomTabNavigator();

export function BottomTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarStyle: {
          height: 120,
          paddingTop: 12,
        },
        tabBarShowLabel: false,
        tabBarActiveTintColor: "#517861",
        tabBarInactiveTintColor: "#808080",
        tabBarIcon: ({ focused, color }) => {
          let iconName: keyof typeof MaterialCommunityIcons.glyphMap = "home";

          if (route.name === "Home") iconName = "home";
          else if (route.name === "Scan") iconName = "crop-free";
          else if (route.name === "Recents") iconName = "history";
          else if (route.name === "User") iconName = "account";

          return (
            <View
              style={{
                width: 48,
                height: 48,
                borderRadius: 24,
                backgroundColor: focused ? "#6060604a" : "transparent",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <MaterialCommunityIcons name={iconName} size={28} color={color} />
            </View>
          );
        },
      })}
    >
      <Tab.Screen name="Home" component={Home} />
      <Tab.Screen
        name="Scan"
        component={ScanStack}
        options={{ headerShown: false }}
      />
      <Tab.Screen name="Recents" component={Home} />
      <Tab.Screen name="User" component={Home} />
    </Tab.Navigator>
  );
}
