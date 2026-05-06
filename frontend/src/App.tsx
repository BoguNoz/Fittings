import {compositeStore} from "./stores/composite-store.ts";
import {appComposites} from "./repositories/composites.ts";
import Form from "./components/layout/FormSidePanel.tsx";
import {formStore} from "./stores/form-store.ts";
import Dashboard from "./components/layout/Dashboard.tsx";

const App = () => {

    compositeStore.initializeComposite(appComposites)

  return (
      <div className="flex min-h-screen items-center justify-center">
          <Dashboard
              compositeStore={compositeStore}
              formStore={formStore}
          />
      </div>

  )
}

export default App
